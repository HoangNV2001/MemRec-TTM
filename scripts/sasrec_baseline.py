"""
SASRec reference ranker on a fixed temporal cohort (CPU is enough at this scale)

Training sequences = every user's strictly-past history in the cohort's dataset (cohort users: everything
before their target, i.e. train + warm-up item). Early stopping on NDCG@5 of the cohort's *val* users over
their fixed candidates; the test split is scored once with the best epoch.

    python scripts/sasrec_baseline.py --config configs/runs/ml_popmatched_memrec.yaml --prefix ml_popmatched
    python scripts/sasrec_baseline.py ... --loss ce --tag sasrecce --device cuda   # full-softmax cross-entropy

--loss bce (original SASRec): one uniform negative per position. --loss ce: softmax over the whole catalog at every
position, which is known to train SASRec much better (Klenitskiy & Vasilev, RecSys 2023).
Cohorts without `holdout_cohort_only` (InstructRec) use the upstream leave-one-out split for every user.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RecDataset  # noqa: E402
from src.eval.config import load_run_config  # noqa: E402
from src.eval.stats import metrics_from_positions  # noqa: E402


class SASRec(nn.Module):
    """Causal self-attention next-item model; item 0 is padding (ids shifted by +1)"""

    def __init__(self, n_items, dim=64, max_len=50, blocks=2, heads=2, dropout=0.2):
        super().__init__()
        self.max_len, self.dim = max_len, dim
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation='gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(layer, blocks)
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def encode(self, seq):
        b, n = seq.shape
        pad = seq.eq(0)
        h = self.item(seq) * self.dim ** 0.5 + self.pos(torch.arange(n, device=seq.device).unsqueeze(0).expand(b, -1))
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=seq.device), diagonal=1)
        h = self.enc(self.drop(h), mask=causal, src_key_padding_mask=pad)
        return self.norm(h).masked_fill(pad.unsqueeze(-1), 0.0)


def left_pad(seqs, max_len):
    """Right-aligned (newest last) so the final position is always the latest item"""
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for r, s in enumerate(seqs):
        s = s[-max_len:]
        out[r, max_len - len(s):] = np.asarray(s) + 1
    return torch.from_numpy(out)


def score(model, histories, candidates, max_len):
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        h = model.encode(left_pad(histories, max_len).to(dev))[:, -1]  # newest position
        c = model.item(torch.as_tensor(np.asarray(candidates) + 1, device=dev))
        return (h.unsqueeze(1) * c).sum(-1).cpu().numpy()


def evaluate(model, users, hist_of, max_len):
    if not users:
        return [], []
    s = score(model, [hist_of(u['user_id']) for u in users], [u['candidates'] for u in users], max_len)
    positions, ranked_all = [], []
    for u, row in zip(users, s):
        order = sorted(range(len(u['candidates'])), key=lambda k: (-row[k], k))
        ranked = [u['candidates'][k] for k in order]
        ranked_all.append((ranked, {str(c): float(row[k]) for k, c in enumerate(u['candidates'])}))
        positions.append(ranked.index(u['target']))
    return positions, ranked_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--max_len', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dump_topk', type=int, default=0, help='also write full-catalog top-N lists for cohort users')
    ap.add_argument('--device', default='cpu', help='cpu or cuda')
    ap.add_argument('--loss', default='bce', choices=['bce', 'ce'])
    ap.add_argument('--tag', default='sasrec', help='run dir = results/runs/<prefix>_<tag>')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    torch.set_num_threads(12)

    cfg = load_run_config(args.config)
    cohort = json.load(open(PROJECT_ROOT / cfg['cohort']))
    holdout = {u['user_id'] for u in cohort['users']}
    ds = RecDataset(str(PROJECT_ROOT / cfg['data_dir'] / f"{cfg['dataset']}.inter"),
                    holdout_users=holdout if cohort.get('holdout_cohort_only') else None)

    def hist_of(u):
        return ds.get_user_history(u, 'valid') if u in holdout else ds.train_data[u]

    seqs = [hist_of(u) for u in ds.train_data]
    seqs = [s for s in seqs if len(s) >= 2]
    n_items = int(ds.n_items)
    print(f'{len(seqs)} training sequences, {n_items} items')
    dev = torch.device(args.device)
    model = SASRec(n_items, args.dim, args.max_len).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    val = [u for u in cohort['users'] if u['split'] == 'val']
    test = [u for u in cohort['users'] if u['split'] == 'test']

    # (input, target) pairs: input = s[:-1], target = s[1:], right-aligned windows of max_len + 1
    windows = [s[-(args.max_len + 1):] for s in seqs]
    best, best_state, bad, best_ep = -1.0, None, 0, args.epochs
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        perm = rng.permutation(len(windows))
        for b in range(0, len(perm), args.batch):
            batch = [windows[k] for k in perm[b:b + args.batch]]
            inp = left_pad([w[:-1] for w in batch], args.max_len).to(dev)
            pos = left_pad([w[1:] for w in batch], args.max_len).to(dev)
            neg = torch.randint(1, n_items + 1, pos.shape, device=dev) * pos.ne(0)
            h = model.encode(inp)
            mask = pos.ne(0)
            if args.loss == 'ce':
                logits = h[mask] @ model.item.weight[1:].T  # (positions, n_items), padding row excluded
                loss = F.cross_entropy(logits, pos[mask] - 1)
            else:
                pl = (h * model.item(pos)).sum(-1)[mask]
                nl = (h * model.item(neg)).sum(-1)[mask]
                loss = F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl)) + \
                    F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if not val:  # cohort without a val split (test extension): train exactly --epochs, keep the last state
            print(f'epoch {ep + 1} loss {tot:.1f} (no val split, {time.time() - t0:.0f}s)', flush=True)
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
            continue
        pv, _ = evaluate(model, val, hist_of, args.max_len)
        v = metrics_from_positions(pv)['NDCG@5']
        print(f'epoch {ep + 1} loss {tot:.1f} val NDCG@5 {v:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if v > best:
            best, bad, best_ep = v, 0, ep + 1
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    out = PROJECT_ROOT / 'results' / 'runs' / f'{args.prefix}_{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    recs, summary = [], {}
    for split, users in (('test', test), ('val', val)):
        if not users:
            continue
        pos, ranked = evaluate(model, users, hist_of, args.max_len)
        summary[split] = metrics_from_positions(pos)
        for u, p, (r, sc) in zip(users, pos, ranked):
            recs.append({'user_id': u['user_id'], 'split': split, 'target': u['target'], 'candidates': u['candidates'],
                         'ranked': r, 'position': p, 'scores': sc, 'n_missing': 0})
    with open(out / 'predictions.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    (out / 'summary.json').write_text(json.dumps({'metrics': summary, 'best_val_ndcg5': best, 'best_epoch': best_ep,
                                                  'args': vars(args)}, indent=1))
    if args.dump_topk:
        # full-catalog ranking for stage-1 retrieval (history items excluded)
        model.eval()
        all_items = model.item.weight[1:].detach()
        topk = {}
        with torch.no_grad():
            for b in range(0, len(cohort['users']), 256):
                batch = cohort['users'][b:b + 256]
                h = model.encode(left_pad([hist_of(u['user_id']) for u in batch], args.max_len).to(dev))[:, -1]
                sc = (h @ all_items.T).cpu().numpy()
                for u, row in zip(batch, sc):
                    row[list(set(hist_of(u['user_id'])))] = -np.inf
                    idx = np.argpartition(-row, args.dump_topk)[:args.dump_topk]
                    topk[str(u['user_id'])] = [int(i) for i in idx[np.argsort(-row[idx])]]
        json.dump({'n': args.dump_topk, 'topk': topk}, open(out / 'topk.json', 'w'))
        print('->', out / 'topk.json')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
