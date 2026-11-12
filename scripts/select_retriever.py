"""
Choose the stage-1 fusion on validation users only: pooled Recall@K over several retrieval_<prefix>.json files

    python scripts/select_retriever.py --files ml_val ml_val2 amz_val amz_val2 --k 50
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--top', type=int, default=12)
    args = ap.parse_args()

    per = [json.load(open(PROJECT_ROOT / 'results' / 'explore' / f'retrieval_{f}.json'))['per_user_pos']
           for f in args.files]
    names = set.intersection(*[set(p) for p in per])
    rows = []
    for name in names:
        hits = [[0 <= v < args.k for v in p[name].values()] for p in per]
        pooled = float(np.mean(np.concatenate(hits)))
        rows.append((pooled, name, [float(np.mean(h)) for h in hits]))
    print(f'pooled Recall@{args.k} over {", ".join(args.files)} (n = {sum(len(p["ppr"]) for p in per)})')
    for pooled, name, each in sorted(rows, reverse=True)[:args.top]:
        print(f'  {name:32s} {pooled:.3f}   ' + '  '.join(f'{x:.3f}' for x in each))


if __name__ == '__main__':
    sys.exit(main())
