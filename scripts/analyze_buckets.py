"""
RQ4 bucket analysis: paired NDCG@5 of several runs split by user/target properties

    python scripts/analyze_buckets.py --config configs/runs/ml_popmatched_memrec.yaml \
        --runs ml_pm_memrec_t300 ml_pm_ttm_full_t300 --graph ml_popmatched_ppr

Buckets (terciles on the test users): history length, gold popularity (past interactions),
graph rank of the gold among the 10 candidates (from the graph run's scores).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RecDataset  # noqa: E402
from src.eval.config import load_run_config  # noqa: E402
from src.eval.stats import load_predictions, ndcg_at  # noqa: E402


def terciles(values):
    q = np.quantile(values, [1 / 3, 2 / 3])
    return np.digitize(values, q), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--graph', required=True)
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    cohort = json.load(open(PROJECT_ROOT / cfg['cohort']))
    holdout = [u['user_id'] for u in cohort['users']]
    ds = RecDataset(str(PROJECT_ROOT / cfg['data_dir'] / f"{cfg['dataset']}.inter"), holdout_users=holdout)
    pop = np.zeros(ds.n_items)
    for items in ds.train_data.values():
        np.add.at(pop, items, 1)

    runs = {r: load_predictions(PROJECT_ROOT / 'results' / 'runs' / r / 'predictions.jsonl') for r in args.runs}
    G = load_predictions(PROJECT_ROOT / 'results' / 'runs' / args.graph / 'predictions.jsonl')
    users = sorted(u for u in G if G[u]['split'] == args.split and all(u in R for R in runs.values()))

    feats = {
        'history_len': np.array([len(ds.get_user_history(u, 'valid')) for u in users]),
        'gold_pop': np.array([pop[G[u]['target']] for u in users]),
        'gold_graph_rank': np.array([G[u]['position'] for u in users]),
    }
    nd = {r: np.array([ndcg_at(R[u]['position'], 5) for u in users]) for r, R in runs.items()}
    out = {'n': len(users)}
    for name, v in feats.items():
        b, q = terciles(v) if name != 'gold_graph_rank' else (np.digitize(v, [1, 3]), [1, 3])
        rows = []
        for k in range(3):
            m = b == k
            rows.append({'bucket': k, 'n': int(m.sum()), 'range': [float(v[m].min()), float(v[m].max())] if m.any() else None,
                         **{r: float(nd[r][m].mean()) for r in runs}})
        out[name] = {'cuts': [float(x) for x in q], 'rows': rows}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
