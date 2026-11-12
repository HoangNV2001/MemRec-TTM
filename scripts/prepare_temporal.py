"""
Prepare a temporal dataset (Amazon Books Kaggle / MovieLens-32M) in MemRec format + fixed cohorts

Protocol (see docs/PLAN.md §4)
- Global cutoff c = quantile(q) of event timestamps. Past = events with ts < c.
- Cohort user: >= min_hist past events; target = the user's first event with ts >= c, which must be
  the only event of that user at that timestamp, a novel item for the user, and an item seen in the past
  (candidate pool = items with >= 1 past event).
- .inter = all past events of the kept users + the target event of cohort users. With
  `holdout_cohort_only`, RecDataset leave-one-out applies to cohort users only, so test = target,
  valid = last past event (used for warm-up), and everyone else's past stays in the graph.
- Candidates: target + n_neg negatives the user never interacted with (whole raw data), order shuffled
  deterministically per (seed, user). Two schemes over the SAME users:
    uniform     negatives uniform over the pool
    popmatched  negatives from the gold item's log2-popularity bin (past counts), widened if the bin is small
  Warm-up target (last past event) gets candidates by the same scheme.

    python scripts/prepare_temporal.py --dataset amazon_books
    python scripts/prepare_temporal.py --dataset ml32m --sample_users 20000
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'data_downloaded'


def _clean_list(s):
    """"['A', 'B']" -> "A, B" """
    if not isinstance(s, str) or not s.strip():
        return ''
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return ', '.join(str(x) for x in v)
    except (ValueError, SyntaxError):
        pass
    return s


def load_amazon_books():
    ev = pd.read_csv(RAW / 'amazon_books/raw/Books_rating.csv',
                     usecols=['Title', 'User_id', 'review/time'])
    ev = ev.dropna(subset=['Title', 'User_id', 'review/time'])
    ev['ts'] = pd.to_numeric(ev['review/time'], errors='coerce')
    ev = ev.dropna(subset=['ts'])
    ev = ev.rename(columns={'User_id': 'user', 'Title': 'item'})[['user', 'item', 'ts']]
    ev['ts'] = ev['ts'].astype('int64')
    meta = pd.read_csv(RAW / 'amazon_books/raw/books_data.csv',
                       usecols=['Title', 'description', 'authors', 'categories'])
    meta = meta.dropna(subset=['Title']).drop_duplicates('Title').set_index('Title')

    def describe(title):
        if title not in meta.index:
            return ''
        r = meta.loc[title]
        parts = []
        a, c = _clean_list(r['authors']), _clean_list(r['categories'])
        if a:
            parts.append(f'By {a}.')
        if c:
            parts.append(f'Genre: {c}.')
        if isinstance(r['description'], str) and r['description'].strip():
            parts.append(r['description'].strip())
        return ' '.join(parts)

    return ev, (lambda item: str(item)), describe, 'instructrec-books'


def load_ml32m():
    ev = pd.read_csv(RAW / 'ml-32m/ratings.csv')
    ev = ev[ev['rating'] >= 4.0].rename(columns={'userId': 'user', 'movieId': 'item', 'timestamp': 'ts'})
    ev = ev[['user', 'item', 'ts']]
    movies = pd.read_csv(RAW / 'ml-32m/movies.csv').set_index('movieId')
    tags = pd.read_csv(RAW / 'ml-32m/tags.csv', usecols=['movieId', 'tag']).dropna()
    tags['tag'] = tags['tag'].astype(str).str.strip().str.lower()
    top_tags = (tags.groupby(['movieId', 'tag']).size().reset_index(name='n')
                .sort_values(['movieId', 'n'], ascending=[True, False])
                .groupby('movieId')['tag'].apply(lambda s: ', '.join(s.head(8))))

    def title(item):
        return str(movies.loc[item, 'title']) if item in movies.index else f'Movie {item}'

    def describe(item):
        parts = []
        if item in movies.index:
            g = str(movies.loc[item, 'genres'])
            if g and g != '(no genres listed)':
                parts.append(f"Genres: {g.replace('|', ', ')}.")
        if item in top_tags.index:
            parts.append(f'Tags: {top_tags.loc[item]}.')
        return ' '.join(parts)

    return ev, title, describe, 'instructrec-movietv'


def sample_negatives(rng, pool_items, exclude, n):
    negs = []
    tries = 0
    while len(negs) < n:
        x = int(pool_items[rng.randint(len(pool_items))])
        tries += 1
        if x not in exclude and x not in negs:
            negs.append(x)
        if tries > 100000:
            raise RuntimeError('negative sampling stuck')
    return negs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['amazon_books', 'ml32m'], required=True)
    ap.add_argument('--cutoff_q', type=float, default=0.9)
    ap.add_argument('--min_hist', type=int, default=5)
    ap.add_argument('--n_test', type=int, default=300)
    ap.add_argument('--n_val', type=int, default=100)
    ap.add_argument('--n_neg', type=int, default=9)
    ap.add_argument('--sample_users', type=int, default=None, help='keep cohort + this many other users')
    ap.add_argument('--min_bin_items', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip', type=int, default=0,
                    help='skip the first N users of the cohort permutation (disjoint extra cohorts, e.g. --skip 400)')
    ap.add_argument('--tag', default=None, help='suffix of the dataset name, e.g. val2 -> temporal-ml32m-val2')
    ap.add_argument('--target_max_q', type=float, default=None,
                    help='training world: keep targets strictly before this time quantile (the test cutoff, e.g. 0.9)')
    ap.add_argument('--exclude_q', type=float, default=None,
                    help='training world: recompute the cohort of the world with this cutoff and exclude its users')
    ap.add_argument('--exclude_n', type=int, default=0, help='how many users of that cohort permutation to exclude')
    ap.add_argument('--split_names', default='test,val', help='names of the two splits, e.g. train,val')
    ap.add_argument('--exclude_users_file', default=None,
                    help='raw user ids (one per line) to skip, e.g. the RL training world; the permutation is kept, '
                         'excluded users are dropped from the positions after --skip')
    ap.add_argument('--extend_to', type=int, default=None,
                    help='also write <scheme><N> cohorts: same users/targets/warm-up, test candidates padded to N')
    args = ap.parse_args()

    name = {'amazon_books': 'temporal-amazon-books', 'ml32m': 'temporal-ml32m'}[args.dataset]
    if args.tag:
        name = f'{name}-{args.tag}'
    ev, title_of, describe, rules = {'amazon_books': load_amazon_books, 'ml32m': load_ml32m}[args.dataset]()
    # one event per (user, item): keep the earliest
    ev = ev.sort_values(['user', 'ts', 'item'], kind='mergesort').drop_duplicates(['user', 'item'], keep='first')
    cutoff = int(ev['ts'].quantile(args.cutoff_q))
    past, future = ev[ev['ts'] < cutoff], ev[ev['ts'] >= cutoff]
    print(f'{name}: {len(ev)} events, cutoff={cutoff} ({pd.to_datetime(cutoff, unit="s")}), past={len(past)}')

    past_count = past.groupby('item').size()
    pool = set(past_count.index)
    hist_len = past.groupby('user').size()

    # target: first future event of the user, alone at its timestamp, novel, in pool
    first_ts = future.groupby('user')['ts'].transform('min')
    first = future[future['ts'] == first_ts]
    first = first[first.groupby('user')['item'].transform('size') == 1]
    # novelty holds by construction: (user, item) was de-duplicated keeping the earliest event
    first = first[first['user'].isin(hist_len[hist_len >= args.min_hist].index) & first['item'].isin(pool)]
    print(f'eligible cohort users: {len(first)}')

    def eligible_at(q):
        """Same target rule at another cutoff quantile (used to find the users of the test world)"""
        c = int(ev['ts'].quantile(q))
        pa, fu = ev[ev['ts'] < c], ev[ev['ts'] >= c]
        hl = pa.groupby('user').size()
        f = fu[fu['ts'] == fu.groupby('user')['ts'].transform('min')]
        f = f[f.groupby('user')['item'].transform('size') == 1]
        return f[f['user'].isin(hl[hl >= args.min_hist].index) & f['item'].isin(set(pa['item']))]

    if args.target_max_q is not None:  # training labels must come before the test cutoff
        first = first[first['ts'] < int(ev['ts'].quantile(args.target_max_q))]
        print(f'  targets before q={args.target_max_q}: {len(first)}')
    if args.exclude_q is not None and args.exclude_n:
        other = eligible_at(args.exclude_q)
        excluded = set(other.iloc[np.random.RandomState(args.seed).permutation(len(other))[:args.exclude_n]]['user'])
        first = first[~first['user'].isin(excluded)]
        print(f'  excluding {len(excluded)} users of the q={args.exclude_q} cohort: {len(first)} left')

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(first))
    if args.exclude_users_file:  # same permutation as without the option; only the users after --skip are filtered
        drop = {line.strip() for line in open(args.exclude_users_file) if line.strip()}
        rest = [i for i in perm[args.skip:] if str(first['user'].iloc[i]) not in drop]
        print(f'  exclude_users_file: dropped {len(perm) - args.skip - len(rest)} of {len(perm) - args.skip} users after skip')
        chosen = first.iloc[rest[:args.n_test + args.n_val]]
    else:
        chosen = first.iloc[perm[args.skip:args.skip + args.n_test + args.n_val]]
    assert len(chosen) == args.n_test + args.n_val, f'only {len(chosen)} eligible users left'
    cohort_users = chosen['user'].tolist()

    keep_users = set(past['user'].unique())
    if args.sample_users:
        others = sorted(keep_users - set(cohort_users))
        keep_users = set(cohort_users) | set(rng.choice(others, size=min(args.sample_users, len(others)), replace=False).tolist())
    past = past[past['user'].isin(keep_users)]
    # popularity / pool from the data the models actually see
    past_count = past.groupby('item').size()
    pool_items = np.array(sorted(past_count.index, key=str), dtype=object)

    # id maps
    items_sorted = sorted(set(past['item']) | set(chosen['item']), key=str)
    item_id = {it: k for k, it in enumerate(items_sorted)}
    users_sorted = sorted(keep_users, key=str)
    user_id = {u: k for k, u in enumerate(users_sorted)}

    out_dir = ROOT / 'data' / 'processed' / name
    out_dir.mkdir(parents=True, exist_ok=True)
    inter = pd.concat([past, chosen[['user', 'item', 'ts']]])
    inter = pd.DataFrame({'user_id': inter['user'].map(user_id), 'item_id': inter['item'].map(item_id),
                          'rating': 5.0, 'timestamp': inter['ts'].astype('int64')})
    inter = inter.sort_values(['user_id', 'timestamp', 'item_id'], kind='mergesort')
    inter.to_csv(out_dir / f'{name}.inter', sep='\t', index=False)
    meta = pd.DataFrame({'item_id': range(len(items_sorted)),
                         'asin': [str(it) for it in items_sorted],
                         'title': [title_of(it) for it in items_sorted],
                         'description': [describe(it) for it in items_sorted]})
    meta.to_csv(out_dir / f'{name}.meta', sep='\t', index=False)
    print(f'.inter {len(inter)} rows, {len(users_sorted)} users, {len(items_sorted)} items; '
          f'.meta with description: {(meta.description.str.len() > 0).mean():.3f}')

    # negatives exclude everything the user ever touched (raw data)
    user_all = ev[ev['user'].isin(cohort_users)].groupby('user')['item'].apply(set).to_dict()
    pool_ids = np.array([item_id[it] for it in pool_items])
    pop = np.array([past_count[it] for it in pool_items])
    bins = np.floor(np.log2(pop)).astype(int)

    def bin_pool(item_raw, exclude=(), need=0):
        b = int(np.floor(np.log2(max(past_count.get(item_raw, 1), 1))))
        lo, hi = b, b
        sel = pool_ids[(bins >= lo) & (bins <= hi)]
        excl = np.array(sorted(exclude), dtype=np.int64)
        while len(sel) < args.min_bin_items or (need and (~np.isin(sel, excl)).sum() < 3 * need):
            lo, hi = lo - 1, hi + 1
            sel = pool_ids[(bins >= lo) & (bins <= hi)]
        return sel

    past_by_user = past[past['user'].isin(cohort_users)].sort_values(['user', 'ts', 'item'], kind='mergesort') \
        .groupby('user')['item'].apply(list).to_dict()
    cohorts = {s: [] for s in ('uniform', 'popmatched')}
    for k, (u, tgt) in enumerate(zip(chosen['user'], chosen['item'])):
        split = args.split_names.split(',')[0] if k < args.n_test else args.split_names.split(',')[1]
        excl = {item_id[i] for i in user_all[u] if i in item_id}
        warm_raw = past_by_user[u][-1]
        uid = user_id[u]
        for scheme in ('uniform', 'popmatched'):
            rng_u = np.random.RandomState((args.seed * 1000003 + uid * 7 + (scheme == 'popmatched')) % (2**32))
            entry = {'user_id': uid, 'split': split}
            for role, raw in (('test', tgt), ('warm', warm_raw)):
                p = pool_ids if scheme == 'uniform' else bin_pool(raw)
                negs = sample_negatives(rng_u, p, excl, args.n_neg)
                cands = [item_id[raw]] + negs
                rng_u.shuffle(cands)
                if role == 'test':
                    entry.update({'target': item_id[raw], 'candidates': [int(c) for c in cands]})
                else:
                    entry['warmup'] = [{'target': item_id[raw], 'candidates': [int(c) for c in cands]}]
            cohorts[scheme].append(entry)
            if args.extend_to:
                # extra negatives from the same pool, own RNG: the base cohort above is unchanged
                rng_x = np.random.RandomState((args.seed * 7919 + uid * 13 + (scheme == 'popmatched')) % (2**32))
                taken, n_extra = excl | set(entry['candidates']), args.extend_to - len(entry['candidates'])
                p = pool_ids if scheme == 'uniform' else bin_pool(tgt, taken, n_extra)  # heavy users: widen the bin
                extra = sample_negatives(rng_x, p, taken, n_extra)
                cands = entry['candidates'] + extra
                rng_x.shuffle(cands)
                cohorts.setdefault(f'{scheme}{args.extend_to}', []).append({**entry, 'candidates': [int(c) for c in cands]})

    # audit: popularity of gold vs negatives
    id_pop = {item_id[it]: int(c) for it, c in past_count.items()}
    for scheme, users in cohorts.items():
        g = np.array([id_pop.get(e['target'], 0) for e in users])
        n = np.array([[id_pop.get(c, 0) for c in e['candidates'] if c != e['target']] for e in users])
        win = float(np.mean(n < g[:, None]))
        meta_c = {'dataset': name, 'source': args.dataset, 'protocol': f'temporal_global_cutoff_{scheme}',
                  'n_candidates': len(users[0]['candidates']),
                  'holdout_cohort_only': True, 'cutoff': cutoff, 'cutoff_q': args.cutoff_q,
                  'min_hist': args.min_hist, 'n_test': args.n_test, 'n_val': args.n_val, 'seed': args.seed,
                  'domain_rules': rules, 'sample_users': args.sample_users,
                  'audit': {'P(gold_pop > neg_pop)': round(win, 4),
                            'gold_pop_median': float(np.median(g)), 'neg_pop_median': float(np.median(n))}}
        path = ROOT / 'data' / 'cohorts' / f'{name}_{scheme}_t{args.n_test}_v{args.n_val}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({**meta_c, 'users': users}, open(path, 'w'))
        print(scheme, json.dumps(meta_c['audit']), '->', path.name)


if __name__ == '__main__':
    main()
