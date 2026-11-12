"""
CPU headroom check for temporal collaborative neighbors (direction B3, no LLM call)

Sequence-aware user-kNN in the spirit of SKNN/STAN: the neighbors of u are past users who consumed one of u's last m
items; each match contributes the items the neighbor consumed in the next `window` steps after it. Weights:
recency of the matched item in u's sequence (j/m), step distance (1/d), neighbor similarity (sum of match weights,
top `k_users` neighbors), and optionally how recent the neighbor's match is (exp(-age/lambda), STAN-style).
Defaults are the usual literature values, not tuned on these labels.

    python scripts/headroom_sknn.py --config configs/runs/ml_popmatched_memrec.yaml --prefix ml_popmatched
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RecDataset  # noqa: E402
from src.eval.config import load_run_config  # noqa: E402
from src.eval.stats import metrics_from_positions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--m', type=int, default=6)
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--k_users', type=int, default=200)
    ap.add_argument('--lambda_days', type=float, default=365.0)
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    cohort = json.load(open(PROJECT_ROOT / cfg['cohort']))
    holdout = {u['user_id'] for u in cohort['users']}
    ds = RecDataset(str(PROJECT_ROOT / cfg['data_dir'] / f"{cfg['dataset']}.inter"), holdout_users=list(holdout))
    cutoff = cohort.get('cutoff')

    seqs, times = {}, {}
    for u, items in ds.train_data.items():
        if u in holdout:  # cohort users' own past is not a neighbor source for the others' target time
            continue
        seqs[u], times[u] = items, ds.user_times[u][:len(items)]
    index = defaultdict(list)
    for u, items in seqs.items():
        for p, it in enumerate(items):
            index[it].append((u, p))

    def scores(hist, cands, t_now, use_time):
        recent = list(dict.fromkeys(reversed(hist)))[:args.m][::-1]  # distinct, oldest -> newest
        w_rec = {it: (j + 1) / len(recent) for j, it in enumerate(recent)}
        sim, contrib = defaultdict(float), defaultdict(lambda: defaultdict(float))
        hist_set = set(hist)
        for it in recent:
            for v, p in index.get(it, ()):
                w = w_rec[it]
                if use_time and t_now is not None:
                    w *= np.exp(-max(t_now - times[v][p], 0) / (86400 * args.lambda_days))
                sim[v] += w
                for d, nxt in enumerate(seqs[v][p + 1:p + 1 + args.window], 1):
                    if nxt not in hist_set:
                        contrib[v][nxt] += w / d
        top = sorted(sim, key=lambda v: -sim[v])[:args.k_users]
        s = defaultdict(float)
        for v in top:
            for it, c in contrib[v].items():
                s[it] += sim[v] * c
        return np.array([s.get(c, 0.0) for c in cands])

    report = {}
    for name, use_time in (('sknn', False), ('stan', True)):
        recs = []
        for u in cohort['users']:
            hist = ds.get_user_history(u['user_id'], 'valid')
            cands = u['candidates']
            s = scores(hist, cands, cutoff, use_time)
            order = sorted(range(len(cands)), key=lambda k: (-s[k], k))
            ranked = [cands[k] for k in order]
            recs.append({'user_id': u['user_id'], 'split': u['split'], 'target': u['target'], 'candidates': cands,
                         'ranked': ranked, 'position': ranked.index(u['target']),
                         'scores': {str(c): float(v) for c, v in zip(cands, s)}, 'n_missing': 0,
                         'n_zero': int((s == 0).sum())})
        d = PROJECT_ROOT / 'results' / 'runs' / f'{args.prefix}_{name}'
        d.mkdir(parents=True, exist_ok=True)
        with open(d / 'predictions.jsonl', 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        report[name] = {s: metrics_from_positions([r['position'] for r in recs if r['split'] == s])['NDCG@5']
                        for s in ('test', 'val')}
        report[name]['gold_zero_share'] = float(np.mean([r['scores'][str(r['target'])] == 0 for r in recs]))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
