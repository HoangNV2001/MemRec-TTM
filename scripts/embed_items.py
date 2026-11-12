"""
Dense item embeddings for the semantic retrieval channel (bge-small-en-v1.5, CPU or GPU)

Text = title + description (first `max_chars` chars). Embeddings are cached by raw item key (`asin` column of the .meta
file), so datasets that share items (test cohort and val2) are encoded once. Writes, per dataset directory,
`item_emb_<tag>.npy` (float16, rows aligned with item_id, L2-normalized).

    python scripts/embed_items.py data/processed/temporal-ml32m data/processed/temporal-ml32m-val2
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def item_text(row, max_chars):
    title = str(row['title']) if pd.notna(row['title']) else ''
    desc = str(row['description']) if pd.notna(row['description']) else ''
    return f"{title}. {desc[:max_chars]}" if desc and desc.lower() != 'nan' else title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--model', default='BAAI/bge-small-en-v1.5')
    ap.add_argument('--tag', default='bge-small')
    ap.add_argument('--max_chars', type=int, default=300)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--device', default='cpu', help='cpu or cuda')
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(12)
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = 128

    metas = {}
    for d in args.dirs:
        d = PROJECT_ROOT / d
        meta = pd.read_csv(d / f'{d.name}.meta', sep='\t', dtype={'asin': str})
        metas[d] = meta.sort_values('item_id')
    texts = {}
    for meta in metas.values():
        for _, row in meta.iterrows():
            texts.setdefault(row['asin'], item_text(row, args.max_chars))
    keys = sorted(texts)
    print(f'{len(keys)} unique items to encode', flush=True)
    t0 = time.time()
    emb = model.encode([texts[k] for k in keys], batch_size=args.batch, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True)
    print(f'encoded in {time.time() - t0:.0f}s', flush=True)
    index = {k: i for i, k in enumerate(keys)}
    for d, meta in metas.items():
        rows = np.array([index[a] for a in meta['asin']])
        assert (meta['item_id'].values == np.arange(len(meta))).all(), 'item ids must be 0..n-1'
        out = d / f'item_emb_{args.tag}.npy'
        np.save(out, emb[rows].astype(np.float16))
        print('->', out, emb[rows].shape)


if __name__ == '__main__':
    sys.exit(main())
