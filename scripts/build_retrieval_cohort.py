"""
Cohort for the end-to-end two-stage evaluation (E2): candidates = top-K of the stage-1 retriever, target NOT inserted

Same users, targets and warm-up as the base cohort. The candidate list is the retriever's top-K shuffled with a fixed
per-user seed, so the LLM sees no retrieval order; `retrieval_order` keeps the order for the retriever-only baseline.
If the target is not retrieved, every LLM arm scores 0 for that user (full-catalog evaluation).

    python scripts/build_retrieval_cohort.py --config configs/runs/ml_popmatched_memrec.yaml \
        --topk results/runs/ml_retrieval/topk.json --k 20 --name temporal-ml32m_retrieved20_t300_v100
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.config import load_run_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='run config of the base cohort (users, targets, warm-up)')
    ap.add_argument('--topk', required=True)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--name', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    base = json.load(open(PROJECT_ROOT / cfg['cohort']))
    top = json.load(open(PROJECT_ROOT / args.topk))
    users, hits = [], {'val': [], 'test': []}
    for u in base['users']:
        order = [int(i) for i in top['topk'][str(u['user_id'])][:args.k]]
        cands = list(order)
        np.random.RandomState((args.seed * 7919 + u['user_id']) % (2 ** 32)).shuffle(cands)
        users.append({'user_id': u['user_id'], 'split': u['split'], 'target': u['target'], 'candidates': cands,
                      'retrieval_order': order, 'warmup': u['warmup'],
                      'skip_eval': u['target'] not in order})  # scores 0 for every arm: no LLM call needed
        hits.setdefault(u['split'], []).append(u['target'] in order)
    meta = {k: v for k, v in base.items() if k not in ('users', 'audit')}
    meta.update({'protocol': 'two_stage_retrieved', 'retriever': top['fusion'], 'k': args.k, 'base_cohort': cfg['cohort'],
                 'recall_at_k': {s: float(np.mean(h)) for s, h in hits.items() if h}})
    out = PROJECT_ROOT / 'data' / 'cohorts' / f'{args.name}.json'
    json.dump({**meta, 'users': users}, open(out, 'w'))
    print(meta['recall_at_k'], '->', out)


if __name__ == '__main__':
    sys.exit(main())
