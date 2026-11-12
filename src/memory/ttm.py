"""
Temporal Transition Memory (MemRec-TTM): turn transition-graph scores into textual evidence

- C2 trajectory neighbors (read side, candidate-blind): top items of the user's PPR vector that are not in
  the history; they join the Stage-R neighbor table with a short "next-step" evidence string.
- C3 candidate evidence (rerank side): for every candidate, its global PPR percentile among all items and
  any direct transition from a recent seed item.
Percentiles are over the whole catalog, so evidence does not depend on which negatives were sampled.
"""
from typing import Dict, List, Sequence

import numpy as np

from .transition import TransitionGraph


class TrajectoryRetriever:
    def __init__(self, dataset, mode: str = 'ppr', m: int = 6, restart: float = 0.15):
        self.dataset = dataset
        self.graph = TransitionGraph(dataset)
        self.mode = mode
        self.m = m
        self.restart = restart

    def _title(self, item_id: int, n: int = 60) -> str:
        meta = self.dataset.item_metadata.get(item_id, {}) if self.dataset.item_metadata else {}
        return str(meta.get('title', f'Item-{item_id}'))[:n]

    def user_state(self, history: Sequence[int]) -> Dict:
        seeds = self.graph.seeds_from_history(history, self.m)
        if self.mode == 'ppr':
            vec = self.graph.ppr(seeds, self.restart)
        else:
            vec = self.graph.one_step(seeds)
        vec = vec.copy()
        vec[list(set(history))] = 0.0  # already consumed: not a next step
        positive = np.sort(vec[vec > 0])
        return {'seeds': seeds, 'vec': vec, 'positive_sorted': positive}

    def _top_share(self, state: Dict, score: float) -> float:
        """Share of the catalog with a score >= this one (0.01 = top 1%)"""
        pos = state['positive_sorted']
        n_ge = len(pos) - np.searchsorted(pos, score, side='left')
        return max(n_ge, 1) / self.graph.n_items

    def _direct(self, state: Dict, item_id: int):
        """Strongest direct transition seed -> item (raw count)"""
        best, best_seed = 0.0, None
        for s in state['seeds']:
            w = self.graph.W[s, item_id]
            if w > best:
                best, best_seed = w, s
        return best, best_seed

    def describe(self, state: Dict, item_id: int) -> str:
        score = float(state['vec'][item_id])
        if score <= 0:
            return 'no next-step signal from your recent items'
        share = self._top_share(state, score)
        rank = int(round(share * self.graph.n_items))
        where = (f'#{rank} of {self.graph.n_items:,} items' if rank <= 1000 else f'top {100 * share:.2g}% of items')
        w, seed = self._direct(state, item_id)
        if seed is not None:
            n = int(round(w)) if w >= 1 else 1
            return f"{where} by next-step likelihood; {n} user(s) went directly from '{self._title(seed, 40)}' to it"
        return f'{where} by next-step likelihood (multi-step transition paths only)'

    def trajectory_neighbors(self, state: Dict, k: int) -> List[Dict]:
        vec = state['vec']
        if k <= 0 or not (vec > 0).any():
            return []
        top = np.argsort(-vec)[:k]
        out = []
        for rank, i in enumerate(top):
            if vec[i] <= 0:
                break
            out.append({'type': 'item', 'id': int(i), 'trajectory': True,
                        'score': 2.0 - rank / max(k, 1),  # packed first, inside the same token budget
                        'evidence': self.describe(state, int(i))})
        return out

    def candidate_evidence(self, state: Dict, candidates: Sequence[int]) -> Dict[int, str]:
        return {int(c): self.describe(state, int(c)) for c in candidates}
