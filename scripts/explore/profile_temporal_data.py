"""Profile temporal datasets for suitability (CPU only)."""
import sys, json, time
import numpy as np, pandas as pd
from collections import defaultdict, Counter

ROOT = sys.argv[1]
out = {}

def seq_stats(df, name, ucol, icol, tcol):
    t0 = time.time()
    df = df.sort_values([ucol, tcol], kind='stable')
    n = len(df); nu = df[ucol].nunique(); ni = df[icol].nunique()
    L = df.groupby(ucol).size()
    # same-timestamp ties within user
    dup_t = df.duplicated([ucol, tcol]).mean()
    # popularity concentration
    pop = df[icol].value_counts()
    top1 = pop.iloc[: max(1, int(0.01 * len(pop)))].sum() / n
    # global temporal split 80/10/10 on time
    q80, q90 = df[tcol].quantile([0.8, 0.9]).tolist()
    tr = df[df[tcol] < q80]; te = df[df[tcol] >= q90]
    # transition graph from train: consecutive (prev->next) per user, skip equal timestamps
    trs = tr[[ucol, icol, tcol]].to_numpy()
    succ = defaultdict(Counter)
    for k in range(1, len(trs)):
        if trs[k][0] == trs[k - 1][0] and trs[k][2] > trs[k - 1][2]:
            succ[trs[k - 1][1]][trs[k][1]] += 1
    tr_pop = tr[icol].value_counts()
    # test events: user's first event at/after q90 with >=5 prior events in train; last prior item as seed
    last_hist = tr.groupby(ucol)[icol].apply(list)
    te_first = te.groupby(ucol).head(1)
    rng = np.random.RandomState(0)
    pool = tr_pop.index.to_numpy()
    reach = 0; novel = 0; m = 0; pop_win = 0.0; in_pool = 0
    for u, it in zip(te_first[ucol].to_numpy(), te_first[icol].to_numpy()):
        h = last_hist.get(u)
        if h is None or len(h) < 5:
            continue
        m += 1
        if it in set(h):
            continue
        novel += 1
        if it not in tr_pop.index:
            continue
        in_pool += 1
        if any(it in succ.get(s, {}) for s in h[-6:]):
            reach += 1
        # popularity shortcut: P(gold more popular than a uniform negative)
        negs = pool[rng.randint(0, len(pool), 9)]
        pg = tr_pop.get(it, 0); pn = tr_pop.reindex(negs).fillna(0).to_numpy()
        pop_win += (pn < pg).mean()
        if m >= 20000:
            break
    out[name] = dict(
        interactions=int(n), users=int(nu), items=int(ni),
        density=n / (nu * ni),
        user_len_q=L.quantile([.25, .5, .75, .9]).round(1).tolist(),
        frac_users_ge5=float((L >= 5).mean()), frac_users_ge10=float((L >= 10).mean()),
        same_user_timestamp_dup_rate=float(dup_t),
        top1pct_items_share=float(top1),
        train_transition_sources=len(succ), train_transition_pairs=int(sum(len(v) for v in succ.values())),
        test_users_with_ge5_hist=m, novel_targets=novel, novel_in_train_pool=in_pool,
        one_step_reach_from_last6=reach / max(in_pool, 1),
        pop_beats_uniform_negative=pop_win / max(in_pool, 1),
        seconds=round(time.time() - t0, 1),
    )
    print(name, json.dumps(out[name], indent=1), flush=True)

which = sys.argv[2:] or ['ml', 'amz']
if 'ml' in which:
    df = pd.read_csv(f'{ROOT}/ml-32m/ratings.csv')
    df = df[df.rating >= 4.0]
    seq_stats(df, 'ml32m_pos4', 'userId', 'movieId', 'timestamp')
if 'amz' in which:
    df = pd.read_csv(f'{ROOT}/amazon_books/raw/Books_rating.csv',
                     usecols=['Id', 'Title', 'User_id', 'review/score', 'review/time'])
    df = df.dropna(subset=['User_id', 'review/time'])
    df = df[pd.to_numeric(df['review/time'], errors='coerce').notna()]
    df['review/time'] = df['review/time'].astype('int64')
    # item identity = Title (Ids are editions; metadata joins on Title)
    df = df.drop_duplicates(['User_id', 'Title'])
    seq_stats(df, 'amazon_books_title', 'User_id', 'Title', 'review/time')
    meta = pd.read_csv(f'{ROOT}/amazon_books/raw/books_data.csv', usecols=['Title', 'description', 'authors', 'categories'])
    out['amazon_meta'] = dict(rows=len(meta), desc_nonempty=float(meta.description.notna().mean()),
                              cat_nonempty=float(meta.categories.notna().mean()), auth_nonempty=float(meta.authors.notna().mean()))
    print(out['amazon_meta'])
json.dump(out, open(sys.argv[0].replace('.py', f"_{'_'.join(which)}.json"), 'w'), indent=1)
