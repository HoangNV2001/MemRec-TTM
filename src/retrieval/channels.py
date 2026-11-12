"""
Candidate retrieval over the full catalog (stage 1 of the RAG recommender)

Every channel maps a user's strict-past history to a score vector over all items; items already in the history are
excluded. Channels:
- RecentPop   interactions in the `window_days` before the global cutoff (ties: all-time popularity)
- Transition  Personalized PageRank (or one-step) on the temporal transition graph from the m most recent items
- Dense       cosine between item embeddings and the mean embedding of the L most recent items
Channel rankings are fused with reciprocal rank fusion: score(i) = sum_c w_c / (k0 + rank_c(i)).
"""
from typing import Dict, List, Sequence

import numpy as np

from src.memory.transition import TransitionGraph


def _exclude(scores: np.ndarray, history: Sequence[int]) -> np.ndarray:
    s = scores.astype(np.float64, copy=True)
    if len(history):
        s[list(set(history))] = -np.inf
    return s


def top_n(scores: np.ndarray, n: int) -> List[int]:
    n = min(n, int(np.isfinite(scores).sum()))
    idx = np.argpartition(-scores, n - 1)[:n]
    return [int(i) for i in idx[np.argsort(-scores[idx], kind='stable')]]


class RecentPop:
    name = 'pop'

    def __init__(self, dataset, cutoff: int, window_days: float = 90.0):
        n = int(dataset.n_items)
        all_time, recent = np.zeros(n), np.zeros(n)
        lo = cutoff - window_days * 86400
        for u, items in dataset.train_data.items():
            times = dataset.user_times[u][:len(items)]
            np.add.at(all_time, items, 1)
            sel = [i for i, t in zip(items, times) if lo <= t < cutoff]
            if sel:
                np.add.at(recent, sel, 1)
        self.base = recent + all_time / (all_time.max() + 1.0)  # all-time popularity only breaks ties

    def scores(self, history: Sequence[int]) -> np.ndarray:
        return _exclude(self.base, history)


class Transition:
    def __init__(self, dataset, mode: str = 'ppr', m: int = 6, restart: float = 0.15, graph: TransitionGraph = None):
        self.graph = graph or TransitionGraph(dataset)
        self.mode, self.m, self.restart = mode, m, restart
        self.name = mode

    def scores(self, history: Sequence[int]) -> np.ndarray:
        seeds = self.graph.seeds_from_history(history, self.m)
        vec = self.graph.ppr(seeds, self.restart) if self.mode == 'ppr' else self.graph.one_step(seeds)
        return _exclude(vec, history)


class Dense:
    name = 'dense'

    def __init__(self, embeddings: np.ndarray, recent: int = 10):
        self.emb = embeddings.astype(np.float32)
        self.recent = recent

    def query(self, history: Sequence[int]) -> np.ndarray:
        q = self.emb[list(history[-self.recent:])].mean(axis=0)
        return q / (np.linalg.norm(q) + 1e-12)

    def scores(self, history: Sequence[int]) -> np.ndarray:
        return _exclude(self.emb @ self.query(history), history)


def rrf(rankings: Dict[str, List[int]], weights: Dict[str, float] = None, k0: int = 60, n: int = 100) -> List[int]:
    """Reciprocal rank fusion of several ranked lists (1-based ranks)"""
    fused: Dict[int, float] = {}
    for name, ranked in rankings.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        for r, i in enumerate(ranked, 1):
            fused[i] = fused.get(i, 0.0) + w / (k0 + r)
    return [i for i, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def rank_metrics(ranked: List[int], target: int, ks=(10, 20, 50, 100)) -> Dict[str, float]:
    """Full-catalog metrics for one user with a single relevant item"""
    pos = ranked.index(target) if target in ranked else -1
    out = {f'R@{k}': float(0 <= pos < k) for k in ks}
    out['NDCG@10'] = 1.0 / np.log2(pos + 2) if 0 <= pos < 10 else 0.0
    out['pos'] = pos
    return out
