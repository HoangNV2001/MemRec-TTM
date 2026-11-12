"""
Temporal transition graph over items (MemRec-TTM, component C1)

Directed edge a -> b when a user interacts with b in the time batch right after the batch containing a.
Events of one user at the same timestamp form an unordered batch (no edges inside a batch); each source
spreads one unit of weight over the next batch. Built from dataset.train_data only (strictly past).

Scores for a user are computed from seed items (most recent distinct items of the strict-past history):
- one_step: sum_s w_s P(. | s)
- ppr: personalized PageRank with restart to the seed distribution (= distribution of the endpoint of a
  random walk that stops with probability `restart` at every step), solved by power iteration.
"""
from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp


class TransitionGraph:
    def __init__(self, dataset, half_life_days: float = None, ref_time: float = None):
        """half_life_days: optional exponential time decay of edge weights, 0.5 ** ((ref_time - t_next) / half_life);
        None keeps every transition at weight 1 (as used by MemRec-TTM)"""
        self.n_items = int(dataset.n_items)
        rows, cols, vals = [], [], []
        n_pairs = 0
        for u, items in dataset.train_data.items():
            times = dataset.user_times[u][:len(items)]
            batches, cur, cur_t = [], [], None
            for it, t in zip(items, times):
                if cur and t != cur_t:
                    batches.append(cur)
                    cur = []
                cur.append(it)
                cur_t = t
            if cur:
                batches.append(cur)
            batch_times = []
            if half_life_days:
                t_iter = iter(times)
                for b in batches:
                    t_b = None
                    for _ in b:
                        t_b = next(t_iter)
                    batch_times.append(t_b)
            for j, (prev, nxt) in enumerate(zip(batches[:-1], batches[1:])):
                w = 1.0 / len(nxt)
                if half_life_days:
                    w *= 0.5 ** (max(ref_time - batch_times[j + 1], 0.0) / (86400.0 * half_life_days))
                for a in prev:
                    for b in nxt:
                        if a != b:
                            rows.append(a)
                            cols.append(b)
                            vals.append(w)
                n_pairs += 1
        W = sp.csr_matrix((vals, (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float64)
        W.sum_duplicates()
        self.W = W  # raw transition counts
        out = np.asarray(W.sum(axis=1)).ravel()
        self.out_degree = out
        inv = np.divide(1.0, out, out=np.zeros_like(out), where=out > 0)
        self.P = sp.diags(inv) @ W  # row-stochastic on non-dangling rows
        self.PT = self.P.T.tocsr()
        self.n_batch_pairs = n_pairs

    def stats(self) -> Dict:
        return {'n_items': self.n_items, 'n_edges': int(self.W.nnz), 'n_batch_pairs': self.n_batch_pairs,
                'n_sources': int((self.out_degree > 0).sum())}

    # ------------------------------------------------------------------ seeds
    @staticmethod
    def seeds_from_history(history: Sequence[int], m: int = 6) -> List[int]:
        """Most recent m distinct items (newest last)"""
        seen, seeds = set(), []
        for it in reversed(history):
            if it not in seen:
                seeds.append(it)
                seen.add(it)
            if len(seeds) == m:
                break
        return seeds[::-1]

    def _seed_vector(self, seeds: Sequence[int]) -> np.ndarray:
        p = np.zeros(self.n_items)
        if seeds:
            p[list(seeds)] = 1.0 / len(seeds)
        return p

    # ----------------------------------------------------------------- scores
    def one_step(self, seeds: Sequence[int]) -> np.ndarray:
        return self.PT @ self._seed_vector(seeds)

    def ppr(self, seeds: Sequence[int], restart: float = 0.15, iters: int = 40, tol: float = 1e-9) -> np.ndarray:
        """Stationary PPR vector; mass of dangling items returns to the seeds"""
        p = self._seed_vector(seeds)
        if p.sum() == 0:
            return p
        dangling = self.out_degree == 0
        x = p.copy()
        for _ in range(iters):
            moved = self.PT @ x
            lost = x[dangling].sum()
            x_new = restart * p + (1 - restart) * (moved + lost * p)
            if np.abs(x_new - x).sum() < tol:
                x = x_new
                break
            x = x_new
        return x

    def candidate_scores(self, history: Sequence[int], candidates: Sequence[int], mode: str = 'ppr',
                         m: int = 6, restart: float = 0.15, exclude_seeds: bool = True) -> Dict[int, float]:
        seeds = self.seeds_from_history(history, m)
        vec = self.ppr(seeds, restart) if mode == 'ppr' else self.one_step(seeds)
        scores = {int(c): float(vec[c]) for c in candidates}
        if exclude_seeds:  # restart mass on seeds is not evidence for a candidate
            for s in seeds:
                if s in scores:
                    scores[s] = 0.0
        return scores
