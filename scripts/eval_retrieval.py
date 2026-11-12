"""
E1: full-catalog evaluation of the stage-1 retrieval channels and their RRF fusions (CPU only)

    python scripts/eval_retrieval.py --config configs/runs/ml_popmatched_memrec.yaml --splits val,test \
        --prefix ml --dump ppr+dense+pop90 --dump_n 50

Each user's target is the first post-cutoff interaction; items in the strict-past history are excluded. Reports
Recall@{10,20,50,100} and NDCG@10 per split, and optionally writes the fused top-N lists used to build the
retrieved-candidate cohort (results/runs/<prefix>_retrieval/topk.json).
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RecDataset  # noqa: E402
from src.eval.config import load_run_config  # noqa: E402
from src.memory.transition import TransitionGraph  # noqa: E402
from src.retrieval.channels import Dense, RecentPop, Transition, rank_metrics, rrf, top_n  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--splits', default='val,test')
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--emb', default='bge-small')
    ap.add_argument('--windows', default='30,90,365')
    ap.add_argument('--half_lives', default='90,365', help='time-decayed transition graphs (days); empty to skip')
    ap.add_argument('--depth', type=int, default=500, help='per-channel list length fed to RRF')
    ap.add_argument('--sasrec', default=None, help='run dir with a SASRec topk.json to add as a channel')
    ap.add_argument('--dump', default=None, help='fusion name whose top-N lists are written, e.g. ppr+dense+pop90')
    ap.add_argument('--dump_n', type=int, default=50)
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    cohort = json.load(open(PROJECT_ROOT / cfg['cohort']))
    holdout = [u['user_id'] for u in cohort['users']]
    data_dir = PROJECT_ROOT / cfg['data_dir']
    ds = RecDataset(str(data_dir / f"{cfg['dataset']}.inter"), holdout_users=holdout)
    splits = args.splits.split(',')
    users = [u for u in cohort['users'] if u['split'] in splits]

    t0 = time.time()
    graph = TransitionGraph(ds)
    channels = {'ppr': Transition(ds, 'ppr', graph=graph), 'onestep': Transition(ds, 'onestep', graph=graph)}
    for hl in [h for h in args.half_lives.split(',') if h]:
        g = TransitionGraph(ds, half_life_days=float(hl), ref_time=cohort['cutoff'])
        channels[f'ppr_hl{hl}'] = Transition(ds, 'ppr', graph=g)
        channels[f'onestep_hl{hl}'] = Transition(ds, 'onestep', graph=g)
    for w in args.windows.split(','):
        channels[f'pop{w}'] = RecentPop(ds, cohort['cutoff'], float(w))
    emb_path = data_dir / f'item_emb_{args.emb}.npy'
    if emb_path.exists():
        channels['dense'] = Dense(np.load(emb_path))
    sasrec_lists = None
    if args.sasrec:
        sasrec_lists = json.load(open(PROJECT_ROOT / args.sasrec / 'topk.json'))['topk']
    print(f'channels {list(channels)}{" + sasrec" if sasrec_lists else ""} ready in {time.time() - t0:.0f}s', flush=True)

    singles = list(channels)
    pops = [c for c in singles if c.startswith('pop')]
    trans = [c for c in singles if c.startswith(('ppr', 'onestep'))]
    extras = [[], ['dense']] if 'dense' in channels else [[]]
    fusions = []
    for t in trans:
        for extra in extras:
            for p in pops + [None]:
                combo = [t] + extra + ([p] if p else [])
                if len(combo) > 1:
                    fusions.append('+'.join(combo))
    if 'dense' in channels:
        fusions += [f'dense+{p}' for p in pops]
    if sasrec_lists:
        singles.append('sasrec')
        fusions += ['sasrec+' + f for f in list(fusions)] + [f'sasrec+{p}' for p in pops]

    per_user = {name: {} for name in singles + fusions}
    dumped = {}
    t0 = time.time()
    for k, u in enumerate(users):
        hist = ds.get_user_history(u['user_id'], 'valid')
        lists = {name: top_n(ch.scores(hist), args.depth) for name, ch in channels.items()}
        if sasrec_lists:
            lists['sasrec'] = sasrec_lists[str(u['user_id'])][:args.depth]
        for name in singles:
            per_user[name][u['user_id']] = rank_metrics(lists[name], u['target'])
        for name in fusions:
            fused = rrf({c: lists[c] for c in name.split('+')}, n=max(100, args.dump_n))
            per_user[name][u['user_id']] = rank_metrics(fused, u['target'])
            if name == args.dump:
                dumped[u['user_id']] = fused[:args.dump_n]
        if (k + 1) % 100 == 0:
            print(f'  {k + 1}/{len(users)} users ({time.time() - t0:.0f}s)', flush=True)

    report = {'n_items': int(ds.n_items), 'splits': {}}
    for split in splits:
        ids = [u['user_id'] for u in users if u['split'] == split]
        rows = {}
        for name, pu in per_user.items():
            ms = [pu[i] for i in ids]
            rows[name] = {m: float(np.mean([x[m] for x in ms])) for m in ms[0] if m != 'pos'}
        report['splits'][split] = {'n': len(ids), 'results': rows}
    out = PROJECT_ROOT / 'results' / 'explore' / f'retrieval_{args.prefix}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({**report, 'per_user_pos': {n: {str(i): v['pos'] for i, v in pu.items()} for n, pu in per_user.items()}},
              open(out, 'w'))
    for split, r in report['splits'].items():
        print(f"\n== {args.prefix} {split} (n={r['n']})  R@10   R@20   R@50   R@100  NDCG@10")
        for name, m in sorted(r['results'].items(), key=lambda kv: -kv[1]['R@20']):
            print(f"  {name:28s} {m['R@10']:.3f}  {m['R@20']:.3f}  {m['R@50']:.3f}  {m['R@100']:.3f}  {m['NDCG@10']:.4f}")
    if args.dump:
        d = PROJECT_ROOT / 'results' / 'runs' / f'{args.prefix}_retrieval'
        d.mkdir(parents=True, exist_ok=True)
        json.dump({'fusion': args.dump, 'n': args.dump_n, 'topk': {str(k): v for k, v in dumped.items()}},
                  open(d / 'topk.json', 'w'))
        print('->', d / 'topk.json')


if __name__ == '__main__':
    sys.exit(main())
