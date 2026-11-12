"""
LightGCN reference ranker (He et al., SIGIR 2020) on a fixed temporal cohort, same protocol as sasrec_baseline.py

Training graph = every user's strictly-past history in the cohort's dataset (cohort users: everything before their
target, i.e. train + warm-up item); undirected user-item bipartite graph, symmetric normalisation, mean of the layer
embeddings, BPR loss with one uniform negative per positive edge. Early stopping on NDCG@5 of the cohort's *val* users
over their fixed candidates; the test split is scored once with the best epoch. A cohort without a val split (test
extension) is trained for exactly --epochs (the count early stopping chose on the original world).

    python scripts/lightgcn_baseline.py --config configs/runs/amz_popmatched_memrec.yaml --prefix amz_popmatched
    python scripts/lightgcn_baseline.py --config configs/runs/ext/amzx_memrec.yaml --prefix amzx_popmatched --epochs <best>
    python scripts/lightgcn_baseline.py ... --layers 0 --tag bprmf      # BPR-MF (no propagation)

Cohorts without `holdout_cohort_only` (InstructRec) use the upstream leave-one-out split for every user.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RecDataset  # noqa: E402
from src.eval.config import load_run_config  # noqa: E402
from src.eval.stats import metrics_from_positions  # noqa: E402


def norm_adj(users, items, n_users, n_items, device):
    """D^-1/2 A D^-1/2 of the (U + I) x (U + I) bipartite adjacency, as a coalesced sparse tensor"""
    n = n_users + n_items
    row = np.concatenate([users, items + n_users])
    col = np.concatenate([items + n_users, users])
    deg = np.bincount(row, minlength=n).astype(np.float32)
    dinv = np.zeros(n, dtype=np.float32)
    dinv[deg > 0] = deg[deg > 0] ** -0.5
    val = dinv[row] * dinv[col]
    idx = torch.from_numpy(np.vstack([row, col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(val), (n, n)).coalesce().to(device)


def propagate(adj, emb, layers):
    out, h = emb, emb
    for _ in range(layers):
        h = torch.sparse.mm(adj, h)
        out = out + h
    return out / (layers + 1)


def evaluate(final, n_users, users):
    if not users:
        return [], []
    positions, ranked_all = [], []
    with torch.no_grad():
        for u in users:
            cand = u['candidates']
            s = (final[u['user_id']].unsqueeze(0) * final[n_users + torch.as_tensor(cand)]).sum(-1).cpu().numpy()
            order = sorted(range(len(cand)), key=lambda k: (-s[k], k))
            ranked = [cand[k] for k in order]
            ranked_all.append((ranked, {str(c): float(s[k]) for k, c in enumerate(cand)}))
            positions.append(ranked.index(u['target']))
    return positions, ranked_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--patience', type=int, default=4, help='evaluations without improvement before stopping')
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--reg', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu', help='cpu or cuda')
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--max_batches', type=int, default=0, help='timing smoke test: stop each epoch after N batches')
    ap.add_argument('--tag', default='lightgcn', help='run dir = results/runs/<prefix>_<tag>')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    torch.set_num_threads(args.threads)

    cfg = load_run_config(args.config)
    cohort = json.load(open(PROJECT_ROOT / cfg['cohort']))
    holdout = {u['user_id'] for u in cohort['users']}
    ds = RecDataset(str(PROJECT_ROOT / cfg['data_dir'] / f"{cfg['dataset']}.inter"),
                    holdout_users=holdout if cohort.get('holdout_cohort_only') else None)

    def hist_of(u):
        return ds.get_user_history(u, 'valid') if u in holdout else ds.train_data[u]

    pairs = {(u, i) for u in ds.train_data for i in hist_of(u)}
    pairs = np.array(sorted(pairs), dtype=np.int64)
    eu, ei = pairs[:, 0], pairs[:, 1]
    n_users, n_items = int(ds.n_users), int(ds.n_items)
    print(f'{len(pairs)} edges, {n_users} users, {n_items} items', flush=True)
    dev = torch.device(args.device)
    adj = norm_adj(eu, ei, n_users, n_items, dev)
    emb = torch.nn.Parameter(torch.empty(n_users + n_items, args.dim, device=dev))
    torch.nn.init.normal_(emb, std=0.1)
    opt = torch.optim.Adam([emb], lr=args.lr)
    val = [u for u in cohort['users'] if u['split'] == 'val']
    test = [u for u in cohort['users'] if u['split'] == 'test']
    t_eu, t_ei = torch.from_numpy(eu).to(dev), torch.from_numpy(ei).to(dev)

    best, best_ep, best_state, bad, log = -1.0, 0, None, 0, []
    for ep in range(1, args.epochs + 1):
        t0, tot = time.time(), 0.0
        perm = torch.from_numpy(rng.permutation(len(pairs))).to(dev)
        for nb, b in enumerate(range(0, len(perm), args.batch)):
            if args.max_batches and nb >= args.max_batches:
                break
            idx = perm[b:b + args.batch]
            u, p = t_eu[idx], t_ei[idx]
            n = torch.randint(0, n_items, p.shape, device=dev)
            final = propagate(adj, emb, args.layers)
            fu, fp, fn = final[u], final[n_users + p], final[n_users + n]
            loss = -F.logsigmoid((fu * fp).sum(-1) - (fu * fn).sum(-1)).mean()
            loss = loss + args.reg * 0.5 * (emb[u].pow(2).sum() + emb[n_users + p].pow(2).sum()
                                            + emb[n_users + n].pow(2).sum()) / len(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        msg = f'epoch {ep} loss {tot:.2f} ({time.time() - t0:.0f}s)'
        if args.max_batches:
            print(msg + f' [smoke: {args.max_batches} batches]', flush=True)
            return
        if not val:  # test extension: train exactly --epochs, keep the last state
            print(msg + ' (no val split)', flush=True)
            continue
        if ep % args.eval_every and ep != args.epochs:
            print(msg, flush=True)
            continue
        with torch.no_grad():
            pv, _ = evaluate(propagate(adj, emb, args.layers), n_users, val)
        v = metrics_from_positions(pv)['NDCG@5']
        log.append({'epoch': ep, 'val_ndcg5': v})
        print(msg + f' val NDCG@5 {v:.4f}', flush=True)
        if v > best:
            best, best_ep, bad = v, ep, 0
            best_state = emb.detach().clone()
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is None:
        best_state, best_ep = emb.detach().clone(), args.epochs
    with torch.no_grad():
        final = propagate(adj, best_state, args.layers)
    out = PROJECT_ROOT / 'results' / 'runs' / f'{args.prefix}_{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    recs, summary = [], {}
    for split, users in (('test', test), ('val', val)):
        if not users:
            continue
        pos, ranked = evaluate(final, n_users, users)
        summary[split] = metrics_from_positions(pos)
        for u, p, (r, sc) in zip(users, pos, ranked):
            recs.append({'user_id': u['user_id'], 'split': split, 'target': u['target'], 'candidates': u['candidates'],
                         'ranked': r, 'position': p, 'scores': sc, 'n_missing': 0})
    with open(out / 'predictions.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    (out / 'summary.json').write_text(json.dumps({'metrics': summary, 'best_val_ndcg5': best, 'best_epoch': best_ep,
                                                  'val_log': log, 'args': vars(args)}, indent=1))
    print(json.dumps({'best_epoch': best_ep, **summary}, indent=1))


if __name__ == '__main__':
    main()
