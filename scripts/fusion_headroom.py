"""
Late fusion of an LLM ranking with a graph ranker (no LLM call) + headroom diagnostics (RQ1)

    s_c = 1 / log2(r_c + 1) + alpha * g'_c,  r_c = 1-based LLM rank, g' = min-max of graph scores in C_u
alpha = 0.8 frozen from the earlier Transition-PPR calibration (never tuned on these labels).
If all graph scores of a user are equal the LLM ranking is kept.

    python scripts/fusion_headroom.py --llm results/runs/amz_pm_memrec_t300 --graph results/runs/amz_popmatched_ppr
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.stats import load_predictions, metrics_from_positions, ndcg_at, paired_bootstrap  # noqa: E402


def fuse(llm_rec, graph_rec, alpha):
    cands = llm_rec['candidates']
    g = {int(k): v for k, v in graph_rec['scores'].items()}
    lo, hi = min(g.values()), max(g.values())
    if hi <= lo:
        return list(llm_rec['ranked'])
    rank = {c: k + 1 for k, c in enumerate(llm_rec['ranked'])}
    s = {c: 1.0 / math.log2(rank[c] + 1) + alpha * (g[c] - lo) / (hi - lo) for c in cands}
    return sorted(llm_rec['ranked'], key=lambda c: -s[c])  # stable: LLM order breaks ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llm', required=True, help='run dir with predictions.jsonl')
    ap.add_argument('--graph', required=True, help='run dir with predictions.jsonl (+ scores)')
    ap.add_argument('--alpha', type=float, default=0.8)
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default=None, help='write fused predictions into this run dir')
    args = ap.parse_args()

    L = load_predictions(Path(args.llm) / 'predictions.jsonl')
    G = load_predictions(Path(args.graph) / 'predictions.jsonl')
    users = sorted(u for u in L if u in G and L[u]['split'] == args.split)
    for u in users:
        assert L[u]['candidates'] == G[u]['candidates'], f'user {u}: candidates differ'

    fused = {}
    for u in users:
        ranked = fuse(L[u], G[u], args.alpha)
        fused[u] = {**L[u], 'ranked': ranked, 'position': ranked.index(L[u]['target'])}

    def n5(recs):
        return np.array([ndcg_at(recs[u]['position'], 5) for u in users])

    a_llm, a_graph, a_fused = n5(L), n5(G), n5(fused)
    report = {
        'n': len(users), 'alpha': args.alpha,
        'llm': metrics_from_positions([L[u]['position'] for u in users]),
        'graph': metrics_from_positions([G[u]['position'] for u in users]),
        'fused': metrics_from_positions([fused[u]['position'] for u in users]),
        'fused_vs_llm': paired_bootstrap(a_llm, a_fused),
        'fused_vs_graph': paired_bootstrap(a_graph, a_fused),
        'graph_vs_llm': paired_bootstrap(a_llm, a_graph),
        'oracle_best_of_two_NDCG@5': float(np.maximum(a_llm, a_graph).mean()),
        'complementarity': {
            'llm_hit5_graph_miss': int(((a_llm > 0) & (a_graph == 0)).sum()),
            'graph_hit5_llm_miss': int(((a_graph > 0) & (a_llm == 0)).sum()),
            'corr_ndcg5': float(np.corrcoef(a_llm, a_graph)[0, 1]),
        },
    }
    print(json.dumps(report, indent=1))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'predictions.jsonl', 'w') as f:
            for u in users:
                f.write(json.dumps(fused[u]) + '\n')
        (out / 'summary.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
