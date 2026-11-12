"""
Semantic (dense) retrieval evidence for the reranking prompt

The user query is the mean embedding of the most recent items; every candidate gets its rank among all items by
cosine similarity to that query, as a short text. Ranks are over the whole catalog, so the evidence does not depend
on which other candidates were retrieved or sampled.
"""
from typing import Dict, Sequence

import numpy as np


class SemanticEvidence:
    def __init__(self, emb_path: str, recent: int = 10):
        self.emb = np.load(emb_path).astype(np.float32)
        self.n_items = self.emb.shape[0]
        self.recent = recent

    def user_state(self, history: Sequence[int]) -> Dict:
        q = self.emb[list(history[-self.recent:])].mean(axis=0)
        q /= np.linalg.norm(q) + 1e-12
        scores = self.emb @ q
        scores[list(set(history))] = -np.inf  # already consumed
        finite = np.sort(scores[np.isfinite(scores)])
        return {'scores': scores, 'sorted': finite}

    def describe(self, state: Dict, item_id: int) -> str:
        s = state['scores'][item_id]
        rank = len(state['sorted']) - np.searchsorted(state['sorted'], s, side='right') + 1
        where = (f'#{rank} of {self.n_items:,} items' if rank <= 1000
                 else f'top {100 * rank / self.n_items:.2g}% of items')
        return f'{where} by content similarity to your recent items'

    def candidate_evidence(self, state: Dict, candidates: Sequence[int]) -> Dict[int, str]:
        return {int(c): self.describe(state, int(c)) for c in candidates}
