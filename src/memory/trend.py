"""
Trend evidence for the reranking prompt: rank of each candidate by interactions in the window before the cutoff

Surfaces the recent-popularity retrieval channel of stage 1 to the LLM, so the reranker sees the same signals that
selected the candidates. Ranks are over the whole catalog.
"""
from typing import Dict, Sequence

import numpy as np


class TrendEvidence:
    def __init__(self, dataset, cutoff: float, window_days: float = 90.0):
        n = int(dataset.n_items)
        counts = np.zeros(n)
        lo = cutoff - window_days * 86400
        for u, items in dataset.train_data.items():
            times = dataset.user_times[u][:len(items)]
            sel = [i for i, t in zip(items, times) if lo <= t < cutoff]
            if sel:
                np.add.at(counts, sel, 1)
        self.counts = counts
        self.sorted = np.sort(counts)
        self.n_items = n
        self.window = int(window_days)

    def describe(self, item_id: int) -> str:
        c = self.counts[item_id]
        if c <= 0:
            return f'no interactions from any user in the last {self.window} days'
        rank = self.n_items - np.searchsorted(self.sorted, c, side='right') + 1
        where = f'#{rank} of {self.n_items:,} items' if rank <= 1000 else f'top {100 * rank / self.n_items:.2g}% of items'
        return f'{where} by interactions in the last {self.window} days ({int(c)} interactions)'

    def candidate_evidence(self, candidates: Sequence[int]) -> Dict[int, str]:
        return {int(c): self.describe(int(c)) for c in candidates}
