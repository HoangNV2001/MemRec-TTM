"""
Markdown table of several runs on the same cohort: NDCG@5, Hit@1, MRR and paired NDCG@5 deltas (95% CI)

    python scripts/compare_table.py --base amz_un_memrec_t300 --ref amz_un_vanilla_evid_t300 \
        --runs amz_un_vanilla_t300 amz_un_ttm_evid_recent_t300
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.stats import load_predictions, metrics_from_positions, ndcg_at, paired_bootstrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='run the deltas are taken against')
    ap.add_argument('--ref', default=None, help='optional second reference run')
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    names = [args.base] + ([args.ref] if args.ref else []) + [r for r in args.runs if r not in (args.base, args.ref)]
    P = {n: load_predictions(PROJECT_ROOT / 'results' / 'runs' / n / 'predictions.jsonl') for n in names}
    users = sorted(set.intersection(*[{u for u, r in p.items() if r['split'] == args.split} for p in P.values()]))
    for n, p in P.items():
        for u in users:
            if p[u]['candidates'] != P[args.base][u]['candidates']:
                raise ValueError(f'{n} user {u}: candidates differ from {args.base}')
    nd = {n: np.array([ndcg_at(P[n][u]['position'], 5) for u in users]) for n in names}

    def delta(a, b):
        r = paired_bootstrap(nd[a], nd[b])
        star = '*' if r['ci_low'] > 0 or r['ci_high'] < 0 else ''
        return f"{r['delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]{star}"

    cols = ['run', 'NDCG@5', 'Hit@1', 'MRR', f'Δ vs {args.base}'] + ([f'Δ vs {args.ref}'] if args.ref else [])
    print(f'n = {len(users)} ({args.split})\n')
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * len(cols))
    for n in names:
        m = metrics_from_positions([P[n][u]['position'] for u in users])
        row = [n, f"{m['NDCG@5']:.4f}", f"{m['Hit@1']:.3f}", f"{m['MRR']:.3f}",
               '—' if n == args.base else delta(args.base, n)]
        if args.ref:
            row.append('—' if n == args.ref else delta(args.ref, n))
        print('| ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()
