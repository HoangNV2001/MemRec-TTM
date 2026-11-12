"""
CPU reference rankers on a fixed cohort (no LLM): random, pop, one-step transition, PPR transition

    python scripts/cpu_baselines.py --config configs/runs/amz_pop_memrec.yaml

Writes results/runs/<prefix>_<arm>/predictions.jsonl (same format as run_memrec.py) and prints metrics.
Ties keep the cohort's candidate order (which is a deterministic shuffle), so ties are neutral.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RecDataset  # noqa: E402
from src.eval.config import load_run_config  # noqa: E402
from src.eval.stats import metrics_from_positions  # noqa: E402
from src.memory.transition import TransitionGraph  # noqa: E402


def rank_by(scores, candidates):
    order = sorted(range(len(candidates)), key=lambda k: (-scores[candidates[k]], k))
    return [candidates[k] for k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--prefix', default=None)
    ap.add_argument('--splits', default='test,val')
    ap.add_argument('--m', type=int, default=6, help='seed items')
    ap.add_argument('--restart', type=float, default=0.15)
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    cohort = json.load(open(PROJECT_ROOT / cfg['cohort']))
    holdout = [u['user_id'] for u in cohort['users']] if cohort.get('holdout_cohort_only') else None
    t0 = time.time()
    ds = RecDataset(str(PROJECT_ROOT / cfg['data_dir'] / f"{cfg['dataset']}.inter"), holdout_users=holdout)
    print(f'dataset {time.time() - t0:.0f}s')
    t0 = time.time()
    tg = TransitionGraph(ds)
    print(f'transition graph {time.time() - t0:.0f}s {tg.stats()}')

    pop = np.zeros(ds.n_items)
    for items in ds.train_data.values():
        np.add.at(pop, items, 1)
    rng = np.random.RandomState(cfg.get('seed', 42))
    splits = args.splits.split(',')
    users = [u for u in cohort['users'] if u['split'] in splits]
    prefix = args.prefix or Path(args.config).stem.rsplit('_', 1)[0]

    arms = {'random': [], 'pop': [], 'onestep': [], 'ppr': []}
    coverage = {'onestep': [0, 0], 'ppr': [0, 0]}  # [gold reachable, negatives reachable]
    for u in users:
        uid, cands, tgt = u['user_id'], u['candidates'], u['target']
        hist = ds.get_user_history(uid, 'valid')
        base = {'user_id': uid, 'split': u['split'], 'target': tgt, 'candidates': cands, 'n_missing': 0}
        s_rand = {c: rng.rand() for c in cands}
        s_pop = {c: pop[c] for c in cands}
        s_one = tg.candidate_scores(hist, cands, 'onestep', args.m)
        s_ppr = tg.candidate_scores(hist, cands, 'ppr', args.m, args.restart)
        for arm, sc in (('random', s_rand), ('pop', s_pop), ('onestep', s_one), ('ppr', s_ppr)):
            ranked = rank_by(sc, cands)
            arms[arm].append({**base, 'ranked': ranked, 'position': ranked.index(tgt),
                              'scores': {str(c): float(sc[c]) for c in cands}})
        for arm, sc in (('onestep', s_one), ('ppr', s_ppr)):
            coverage[arm][0] += sc[tgt] > 0
            coverage[arm][1] += sum(sc[c] > 0 for c in cands if c != tgt)

    n = len(users)
    n_neg = sum(len(u['candidates']) - 1 for u in users)
    report = {'graph': tg.stats(),
              'coverage': {a: {'gold': v[0] / n, 'neg': v[1] / n_neg} for a, v in coverage.items()}}
    for arm, recs in arms.items():
        out = PROJECT_ROOT / 'results' / 'runs' / f'{prefix}_{arm}'
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'predictions.jsonl', 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        report[arm] = {s: metrics_from_positions([r['position'] for r in recs if r['split'] == s]) for s in splits}
        (out / 'summary.json').write_text(json.dumps({'metrics': report[arm]}, indent=2))
    print(json.dumps({k: v for k, v in report.items()}, indent=1))
    (PROJECT_ROOT / 'results' / 'runs' / f'{prefix}_cpu_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
