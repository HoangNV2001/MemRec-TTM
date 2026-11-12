"""
Build a fixed evaluation cohort for an InstructRec dataset (MemRec's original benchmark)

- Users: seeded permutation of MemRec's official 1k eval sample -> first n_test = test, next n_val = val
- Eval target = last item of the sequence; candidates = iAgent's original `ranked_lists` (10 items, kept in order)
- Warm-up (rounds=1, like upstream): target = second-to-last item; candidates = target + 9 negatives drawn
  uniformly from interacted items not in the user's sequence, deterministic per (seed, user)

    python scripts/build_cohort_instructrec.py --dataset instructrec-books --pkl booksAll_recagent.pkl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='instructrec-books')
    ap.add_argument('--pkl', default='booksAll_recagent.pkl')
    ap.add_argument('--n_test', type=int, default=300)
    ap.add_argument('--n_val', type=int, default=100)
    ap.add_argument('--n_neg', type=int, default=9)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    data_dir = ROOT / 'data' / 'processed' / args.dataset
    inter = pd.read_csv(data_dir / f'{args.dataset}.inter', sep='\t')
    inter = inter.sort_values(['user_id', 'timestamp'], kind='mergesort')
    seqs = {int(u): g.astype(int).tolist() for u, g in inter.groupby('user_id')['item_id']}
    pool = np.array(sorted(inter['item_id'].unique()))

    df = pd.read_pickle(data_dir / args.pkl)
    ranked = {i: [int(x) for x in rl] for i, rl in enumerate(df['ranked_lists'])}

    sample = json.load(open(ROOT / 'data' / 'eval_user_samples' / f'eval_user_sample_1k_{args.dataset}.json'))['user_ids']
    order = np.random.RandomState(args.seed).permutation(sorted(sample)).tolist()
    chosen = [('test', u) for u in order[:args.n_test]] + [('val', u) for u in order[args.n_test:args.n_test + args.n_val]]

    users, overlap_hist, tgt_pos = [], 0, []
    for split, u in chosen:
        seq = seqs[u]
        target, cands = seq[-1], ranked[u]
        assert target in cands and len(set(cands)) == len(cands), f'user {u}: bad ranked list'
        overlap_hist += len(set(cands) & set(seq[:-1]))
        tgt_pos.append(cands.index(target))
        rng = np.random.RandomState(args.seed * 100003 + u)
        w_target, seen = seq[-2], set(seq)
        negs = []
        while len(negs) < args.n_neg:
            x = int(pool[rng.randint(len(pool))])
            if x not in seen and x not in negs:
                negs.append(x)
        w_cands = [w_target] + negs
        rng.shuffle(w_cands)
        users.append({'user_id': int(u), 'split': split, 'target': int(target), 'candidates': cands,
                      'warmup': [{'target': int(w_target), 'candidates': [int(x) for x in w_cands]}]})

    out = Path(args.out) if args.out else ROOT / 'data' / 'cohorts' / f'{args.dataset}_t{args.n_test}_v{args.n_val}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {'dataset': args.dataset, 'protocol': 'instructrec_original_ranked_lists', 'seed': args.seed,
            'n_test': args.n_test, 'n_val': args.n_val,
            'audit': {'candidate_items_in_history_excl_target': overlap_hist,
                      'target_position_counts': np.bincount(tgt_pos, minlength=10).tolist()}}
    json.dump({**meta, 'users': users}, open(out, 'w'))
    print(json.dumps(meta, indent=1))
    print('->', out)


if __name__ == '__main__':
    main()
