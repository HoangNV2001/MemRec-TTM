"""
CPU headroom check for transition-aware item memory (direction B, no LLM call)

Does "where users come to a candidate from" carry signal beyond the candidate's own content and the PPR score?
Item text = TF-IDF of title + description. For user u (last `recent` items of the strict-past history) and
candidate c:
    content  cos(mean x_recent, x_c)
    predctx  cos(mean x_recent, sum_a W[a, c] x_a over the top-K predecessors a of c)   (transition graph, past only)
Writes predictions in the cpu_baselines format to results/runs/<prefix>_{content,predctx}, so
scripts/fusion_headroom.py can fuse them with an LLM run.

    python scripts/headroom_itemctx.py --config configs/runs/ml_popmatched_memrec.yaml --prefix ml_popmatched
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RecDataset  # noqa: E402
from src.eval.config import load_run_config  # noqa: E402
from src.eval.stats import metrics_from_positions  # noqa: E402
from src.memory.transition import TransitionGraph  # noqa: E402

TOKEN = re.compile(r'[a-z0-9]+')


def tfidf(texts, max_df=0.3, min_df=2):
    vocab, rows, cols = {}, [], []
    for r, t in enumerate(texts):
        for w in set(TOKEN.findall(t.lower())):
            cols.append(vocab.setdefault(w, len(vocab)))
            rows.append(r)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(texts), len(vocab)))
    df = np.asarray((X > 0).sum(0)).ravel()
    keep = np.where((df >= min_df) & (df <= max_df * len(texts)))[0]
    X = X[:, keep] @ sp.diags(np.log(len(texts) / df[keep]))
    norm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    return sp.diags(np.divide(1.0, norm, out=np.zeros_like(norm), where=norm > 0)) @ X


def cos_rows(a, B):
    """cosine between one sparse row a and the rows of B"""
    num = (B @ a.T).toarray().ravel()
    na = np.sqrt(a.multiply(a).sum())
    nb = np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel())
    return num / np.maximum(na * nb, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--recent', type=int, default=10)
    ap.add_argument('--top_pred', type=int, default=30)
    args = ap.parse_args()

    cfg = load_run_config(args.config)
    cohort = json.load(open(PROJECT_ROOT / cfg['cohort']))
    holdout = [u['user_id'] for u in cohort['users']]
    ds = RecDataset(str(PROJECT_ROOT / cfg['data_dir'] / f"{cfg['dataset']}.inter"), holdout_users=holdout)
    ds.load_item_metadata()
    meta = ds.item_metadata
    texts = []
    for i in range(ds.n_items):
        m = meta.get(i, {})
        d = m.get('description', '')
        texts.append(f"{m.get('title', '')} {d if isinstance(d, str) else ''}")
    X = tfidf(texts)
    graph = TransitionGraph(ds)
    WT = graph.W.T.tocsr()  # row c = incoming transition counts

    def pred_profile(c):
        row = WT.getrow(c)
        if row.nnz == 0:
            return None
        k = np.argsort(-row.data)[:args.top_pred]
        return sp.csr_matrix(row.data[k]) @ X[row.indices[k]]

    out = {}
    for name in ('content', 'predctx'):
        out[name] = []
    n_nopred = 0
    for u in cohort['users']:
        hist = ds.get_user_history(u['user_id'], 'valid')
        prof = sp.csr_matrix(X[hist[-args.recent:]].mean(0))
        cands = u['candidates']
        s_content = cos_rows(prof, X[cands])
        preds = [pred_profile(c) for c in cands]
        n_nopred += sum(p is None for p in preds)
        s_pred = np.array([0.0 if p is None else cos_rows(prof, p)[0] for p in preds])
        for name, s in (('content', s_content), ('predctx', s_pred)):
            order = sorted(range(len(cands)), key=lambda k: (-s[k], k))
            ranked = [cands[k] for k in order]
            out[name].append({'user_id': u['user_id'], 'split': u['split'], 'target': u['target'],
                              'candidates': cands, 'ranked': ranked, 'position': ranked.index(u['target']),
                              'scores': {str(c): float(v) for c, v in zip(cands, s)}, 'n_missing': 0})
    report = {'n_candidates_without_predecessor': n_nopred}
    for name, recs in out.items():
        d = PROJECT_ROOT / 'results' / 'runs' / f'{args.prefix}_{name}'
        d.mkdir(parents=True, exist_ok=True)
        with open(d / 'predictions.jsonl', 'w') as f:
            for r in recs:
                f.write(json.dumps(r) + '\n')
        report[name] = {s: metrics_from_positions([r['position'] for r in recs if r['split'] == s])['NDCG@5']
                        for s in ('test', 'val')}
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
