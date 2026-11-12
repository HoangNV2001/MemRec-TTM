"""
List the raw user ids of the RL training world (train + val), so a later test cohort can exclude them.

Replays the selection of
    prepare_temporal.py --cutoff_q 0.8 --target_max_q 0.9 --exclude_q 0.9 --exclude_n 700 \
        --n_test 2000 --n_val 200 --split_names train,val --tag train08
and checks it against data/cohorts/<name>-train08_popmatched_t2000_v200.json (target items must match).

    python scripts/rl_world_users.py --dataset amazon_books
    python scripts/rl_world_users.py --dataset ml32m
writes data/cohorts/<name>-train08_users.txt (one raw user id per line)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_temporal import ROOT, load_amazon_books, load_ml32m


def eligible(ev, q, min_hist):
    c = int(ev['ts'].quantile(q))
    pa, fu = ev[ev['ts'] < c], ev[ev['ts'] >= c]
    hl = pa.groupby('user').size()
    f = fu[fu['ts'] == fu.groupby('user')['ts'].transform('min')]
    f = f[f.groupby('user')['item'].transform('size') == 1]
    return f[f['user'].isin(hl[hl >= min_hist].index) & f['item'].isin(set(pa['item']))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['amazon_books', 'ml32m'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min_hist', type=int, default=5)
    args = ap.parse_args()
    name = {'amazon_books': 'temporal-amazon-books', 'ml32m': 'temporal-ml32m'}[args.dataset]
    ev = {'amazon_books': load_amazon_books, 'ml32m': load_ml32m}[args.dataset]()[0]
    ev = ev.sort_values(['user', 'ts', 'item'], kind='mergesort').drop_duplicates(['user', 'item'], keep='first')

    first = eligible(ev, 0.8, args.min_hist)
    first = first[first['ts'] < int(ev['ts'].quantile(0.9))]
    other = eligible(ev, 0.9, args.min_hist)
    excluded = set(other.iloc[np.random.RandomState(args.seed).permutation(len(other))[:700]]['user'])
    first = first[~first['user'].isin(excluded)]
    chosen = first.iloc[np.random.RandomState(args.seed).permutation(len(first))[:2200]]

    # check: target items of the replayed selection == targets in the saved train08 cohort
    coh = json.load(open(ROOT / 'data' / 'cohorts' / f'{name}-train08_popmatched_t2000_v200.json'))
    meta = pd.read_csv(ROOT / 'data' / 'processed' / f'{name}-train08' / f'{name}-train08.meta', sep='\t',
                       usecols=['item_id', 'asin'], dtype={'asin': str})
    asin = dict(zip(meta['item_id'], meta['asin']))
    saved = Counter(asin[e['target']] for e in coh['users'])
    replay = Counter(str(it) for it in chosen['item'])
    assert len(coh['users']) == len(chosen) and saved == replay, 'replayed RL world differs from the saved cohort'

    out = ROOT / 'data' / 'cohorts' / f'{name}-train08_users.txt'
    out.write_text(''.join(f'{u}\n' for u in chosen['user']))
    print(f'{name}: {len(chosen)} RL-world users (targets match the saved cohort) -> {out.name}')


if __name__ == '__main__':
    main()
