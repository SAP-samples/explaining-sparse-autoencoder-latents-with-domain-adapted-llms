"""
Stage 2 of 2-stage vector DB pipeline.

Reads one shard JSONL file, embeds all chunks with BAAI/bge-m3 (dense + sparse),
writes a partial FAISS index (dense) + sparse vectors (scipy npz) + metadata.

Called by embed_shards_array.sh — one job per shard.

Usage:
    python embed_shard.py --shard_path /fsx/.../shard_0000.jsonl \
                          --output_dir /fsx/.../partials/0000
"""

import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np
import scipy.sparse
import torch
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

MODEL_PATH = "/fsx/i777599/models/bge-m3/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    shard_path = Path(args.shard_path)
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not shard_path.exists():
        log.error("Shard not found: %s", shard_path)
        raise SystemExit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Shard  : %s", shard_path.name)
    log.info("Device : %s", device)

    # FlagEmbedding gives direct access to dense + sparse outputs
    model = BGEM3FlagModel(MODEL_PATH, use_fp16=(device == "cuda"))

    # --- Read all chunks ---
    texts    = []
    metadata = []
    with open(shard_path) as fh:
        for line in tqdm(fh, desc=f"Reading {shard_path.name}", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "")
            if not text:
                continue
            texts.append(text)
            metadata.append({"source": obj["source"], "line": obj["line"]})

    log.info("Chunks to embed: %d", len(texts))

    # --- Embed in batches ---
    dense_vecs   = []
    sparse_rows  = []   # list of dicts {token_id: weight}

    for i in tqdm(range(0, len(texts), args.batch_size), desc="Embedding", unit=" batches"):
        batch = texts[i : i + args.batch_size]
        output = model.encode(
            batch,
            batch_size=args.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs.append(output["dense_vecs"])   # (batch, 1024) float32
        sparse_rows.extend(output["lexical_weights"])  # list of dicts

    dense_matrix = np.vstack(dense_vecs).astype(np.float32)
    # Normalize for cosine similarity
    faiss.normalize_L2(dense_matrix)

    # --- Build FAISS dense index ---
    embed_dim = dense_matrix.shape[1]
    index = faiss.IndexFlatIP(embed_dim)
    index.add(dense_matrix)
    log.info("Dense index: %d vectors, dim=%d", index.ntotal, embed_dim)

    # --- Build sparse matrix (CSR) ---
    # Collect all token ids to determine vocab size
    all_token_ids = set()
    for row in sparse_rows:
        all_token_ids.update(int(k) for k in row.keys())
    vocab_size = max(all_token_ids) + 1 if all_token_ids else 1

    rows_idx, cols_idx, vals = [], [], []
    for i, row in enumerate(sparse_rows):
        for tok_id, weight in row.items():
            rows_idx.append(i)
            cols_idx.append(int(tok_id))
            vals.append(float(weight))

    sparse_matrix = scipy.sparse.csr_matrix(
        (vals, (rows_idx, cols_idx)),
        shape=(len(texts), vocab_size),
        dtype=np.float32,
    )
    log.info("Sparse matrix: %s, nnz=%d", sparse_matrix.shape, sparse_matrix.nnz)

    # --- Save outputs ---
    faiss.write_index(index, str(out_dir / "index.faiss"))
    scipy.sparse.save_npz(str(out_dir / "sparse.npz"), sparse_matrix)
    with open(out_dir / "metadata.jsonl", "w") as fh:
        for entry in metadata:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    config = {
        "n_chunks": len(texts),
        "dense_dim": int(embed_dim),
        "sparse_vocab_size": int(vocab_size),
        "sparse_nnz": int(sparse_matrix.nnz),
    }
    with open(out_dir / "shard_config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    log.info("Done → %s", out_dir)


if __name__ == "__main__":
    main()
