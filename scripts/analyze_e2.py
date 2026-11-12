"""
E2: end-to-end (full-catalog) comparison of the two-stage system against retrieval-only rankers, same users

    python scripts/analyze_e2.py --cohort data/cohorts/temporal-ml32m_retrieved50_t300_v100.json \
        --retrieval ml_test_s --channels pop90 ppr_hl90 sasrec ppr_hl90+pop90 \
        --runs ml_r50_vanilla_t300 ml_r50_memrec_t300 ml_r50_ttm_t300 --base ml_r50_memrec_t300

Positions are 0-based ranks of the target in each system's final list, -1 if absent (counts as a miss). Channel
positions come from scripts/eval_retrieval.py (per_user_pos); LLM runs rerank the retrieved top-K.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.stats import load_predictions, ndcg_at, paired_bootstrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--retrieval', required=True, help='prefix of results/explore/retrieval_<prefix>.json')
    ap.add_argument('--channels', nargs='*', default=[])
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--base', required=True, help='system the deltas are taken against (channel or run name)')
    ap.add_argument('--split', default='test')
    ap.add_argument('--k', type=int, default=10)
    args = ap.parse_args()

    cohort = json.load(open(PROJECT_ROOT / args.cohort))
    users = [u for u in cohort['users'] if u['split'] == args.split]
    ids = [u['user_id'] for u in users]
    retrieved = np.array([u['target'] in u['retrieval_order'] for u in users])
    pos = {}
    per = json.load(open(PROJECT_ROOT / 'results' / 'explore' / f'retrieval_{args.retrieval}.json'))['per_user_pos']
    for ch in args.channels:
        pos[ch] = np.array([per[ch][str(i)] for i in ids])
    pos['retriever (RRF top-K order)'] = np.array([u['retrieval_order'].index(u['target']) if u['target'] in
                                                    u['retrieval_order'] else -1 for u in users])
    for r in args.runs:
        P = load_predictions(PROJECT_ROOT / 'results' / 'runs' / r / 'predictions.jsonl')
        pos[r] = np.array([P[i]['position'] for i in ids])
    nd = {n: np.array([ndcg_at(int(p), args.k) for p in v]) for n, v in pos.items()}
    hit = {n: np.array([0 <= p < args.k for p in v], dtype=float) for n, v in pos.items()}
    print(f'n = {len(ids)} ({args.split}); target retrieved in top-{cohort["k"]}: {retrieved.mean():.3f} '
          f'({retrieved.sum()} users)\n')
    print(f'| System | NDCG@{args.k} | Hit@{args.k} | NDCG@{args.k} if retrieved | Δ NDCG@{args.k} vs {args.base} |')
    print('|---|---|---|---|---|')
    for n in pos:
        if n == args.base:
            d = '—'
        else:
            r = paired_bootstrap(nd[args.base], nd[n])
            star = '*' if r['ci_low'] > 0 or r['ci_high'] < 0 else ''
            d = f"{r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]{star}"
        cond = nd[n][retrieved].mean() if retrieved.any() else float('nan')
        print(f'| {n} | {nd[n].mean():.4f} | {hit[n].mean():.3f} | {cond:.4f} | {d} |')


if __name__ == '__main__':
    sys.exit(main())
