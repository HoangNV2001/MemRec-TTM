"""
Stage-2 decision for the two-stage system, on validation strata only (pre-registered rule, docs/PLAN.md §8)

For every stratum (retrieved cohort, split, runs) compares full-catalog NDCG@10 of: the retriever's own RRF order,
each LLM run, and each LLM run fused with the retriever order by RRF (k0 = 60). Pooled with a stratified bootstrap.

    python scripts/decide_stage2.py \
        --stratum data/cohorts/temporal-ml32m_retrieved50_t300_v100.json:val:ml_r50_ttm_v100=ttm,ml_r50_ttm_trend_v100=ttm_trend \
        --stratum data/cohorts/temporal-ml32m-val2_retrieved50_t0_v300.json:val:mlv2_r50_ttm=ttm,mlv2_r50_ttm_trend=ttm_trend
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.stats import load_predictions, ndcg_at, stratified_bootstrap  # noqa: E402
from src.retrieval.channels import rrf  # noqa: E402


def fused_position(llm_ranked, retrieval_order, target):
    fused = rrf({'llm': llm_ranked, 'retriever': retrieval_order}, n=len(retrieval_order))
    return fused.index(target) if target in fused else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stratum', action='append', required=True,
                    help='cohort.json:split:run=label,run=label (labels must match across strata)')
    ap.add_argument('--k', type=int, default=10)
    args = ap.parse_args()

    scores = {}  # system -> list of per-stratum arrays
    for spec in args.stratum:
        cohort_path, split, runs = spec.split(':')
        cohort = json.load(open(PROJECT_ROOT / cohort_path))
        users = [u for u in cohort['users'] if u['split'] == split]
        ret = [u['retrieval_order'].index(u['target']) if u['target'] in u['retrieval_order'] else -1 for u in users]
        systems = {'retriever (RRF order)': ret}
        for item in runs.split(','):
            run, label = item.split('=')
            P = load_predictions(PROJECT_ROOT / 'results' / 'runs' / run / 'predictions.jsonl')
            systems[label] = [P[u['user_id']]['position'] for u in users]
            systems[f'{label} ⊕ retriever'] = [
                -1 if P[u['user_id']].get('skipped') or u['target'] not in u['retrieval_order']
                else fused_position(P[u['user_id']]['ranked'], u['retrieval_order'], u['target']) for u in users]
        for name, pos in systems.items():
            scores.setdefault(name, []).append(np.array([ndcg_at(int(p), args.k) for p in pos]))
        print(f'{Path(cohort_path).stem}:{split}  n={len(users)}  ' +
              '  '.join(f'{n}={np.mean(scores[n][-1]):.4f}' for n in systems))
    base = scores['retriever (RRF order)']
    print(f'\npooled over {len(args.stratum)} strata (n = {sum(len(a) for a in base)}), NDCG@{args.k}:')
    for name, arrs in scores.items():
        mean = float(np.mean(np.concatenate(arrs)))
        if name == 'retriever (RRF order)':
            print(f'  {name:32s} {mean:.4f}')
            continue
        r = stratified_bootstrap([b - a for a, b in zip(base, arrs)])
        star = '*' if r['ci_low'] > 0 or r['ci_high'] < 0 else ''
        print(f"  {name:32s} {mean:.4f}   Δ vs retriever {r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]{star}")


if __name__ == '__main__':
    sys.exit(main())
