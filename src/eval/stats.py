"""
Ranking metrics and paired statistics on per-user prediction files
"""
import json
import math
from typing import Dict, List

import numpy as np


def ndcg_at(pos: int, k: int) -> float:
    """pos is 0-based rank of the single relevant item"""
    return 1.0 / math.log2(pos + 2) if 0 <= pos < k else 0.0


def metrics_from_positions(positions: List[int], ks=(1, 3, 5, 10)) -> Dict[str, float]:
    p = np.asarray(positions)
    out = {'n': int(len(p))}
    for k in ks:
        out[f'Hit@{k}'] = float(np.mean([0 <= x < k for x in p]))
        out[f'NDCG@{k}'] = float(np.mean([ndcg_at(int(x), k) for x in p]))
    out['MRR'] = float(np.mean([1.0 / (x + 1) if x >= 0 else 0.0 for x in p]))
    return out


def load_predictions(path: str) -> Dict[int, dict]:
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            recs[int(r['user_id'])] = r
    return recs


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Mean of (b - a) with a percentile 95% CI from resampling users"""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {
        'delta': float(d.mean()),
        'ci_low': float(np.percentile(means, 2.5)),
        'ci_high': float(np.percentile(means, 97.5)),
        'improved': int((d > 0).sum()),
        'worsened': int((d < 0).sum()),
        'unchanged': int((d == 0).sum()),
        'n': int(len(d)),
    }


def compare_runs(pred_a: str, pred_b: str, k: int = 5, n_boot: int = 10000) -> Dict:
    """Paired NDCG@k comparison of run B against run A on their common users"""
    A, B = load_predictions(pred_a), load_predictions(pred_b)
    users = sorted(set(A) & set(B))
    for u in users:
        if A[u]['candidates'] != B[u]['candidates']:
            raise ValueError(f'user {u}: candidate lists differ between runs')
    a = np.array([ndcg_at(A[u]['position'], k) for u in users])
    b = np.array([ndcg_at(B[u]['position'], k) for u in users])
    res = paired_bootstrap(a, b, n_boot=n_boot)
    res.update({f'A_NDCG@{k}': float(a.mean()), f'B_NDCG@{k}': float(b.mean())})
    return res


def stratified_bootstrap(deltas: List[np.ndarray], n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Pooled mean of per-user deltas over several strata (dataset x split), resampling users within each stratum;
    strata are weighted by their number of users"""
    rng = np.random.RandomState(seed)
    w = np.array([len(d) for d in deltas], dtype=float)
    w /= w.sum()
    boots = np.zeros(n_boot)
    for d, wi in zip(deltas, w):
        boots += wi * d[rng.randint(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {'delta': float(sum(wi * d.mean() for d, wi in zip(deltas, w))),
            'ci_low': float(np.percentile(boots, 2.5)), 'ci_high': float(np.percentile(boots, 97.5)),
            'n': int(sum(len(d) for d in deltas))}
