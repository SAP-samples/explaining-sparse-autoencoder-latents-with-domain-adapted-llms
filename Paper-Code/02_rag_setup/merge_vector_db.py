"""
Merge all partial FAISS indexes + sparse matrices into a single final index.

Usage:
    python merge_vector_db.py \
        --partial_dir /path/to/partials \
        --output_dir  /path/to/final_db
"""

import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np
import scipy.sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial_dir", type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    args = parser.parse_args()

    partial_dir = Path(args.partial_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    partial_indexes = sorted(partial_dir.glob("*/index.faiss"))
    log.info("Found %d partial indexes in %s", len(partial_indexes), partial_dir)

    if not partial_indexes:
        raise SystemExit("No partial indexes found — did the array job complete?")

    merged_index   = None
    merged_meta    = []
    sparse_parts   = []
    total_dense    = 0

    for idx_path in partial_indexes:
        shard_dir = idx_path.parent

        # Dense
        idx = faiss.read_index(str(idx_path))
        log.info("Merging %s (%d vectors)", shard_dir.name, idx.ntotal)
        total_dense += idx.ntotal
        if merged_index is None:
            merged_index = idx
        else:
            merged_index.merge_from(idx)

        # Metadata
        meta_path = shard_dir / "metadata.jsonl"
        if meta_path.exists():
            with open(meta_path) as fh:
                for line in fh:
                    merged_meta.append(json.loads(line))

        # Sparse
        sparse_path = shard_dir / "sparse.npz"
        if sparse_path.exists():
            sparse_parts.append(scipy.sparse.load_npz(str(sparse_path)))
        else:
            log.warning("No sparse.npz in %s — skipping sparse for this shard", shard_dir.name)

    log.info("Total dense vectors: %d", merged_index.ntotal)

    # Save dense index
    faiss.write_index(merged_index, str(out_dir / "index.faiss"))

    # Merge sparse matrices (stack rows, pad cols to same vocab size)
    if sparse_parts:
        max_cols = max(m.shape[1] for m in sparse_parts)
        padded = []
        for m in sparse_parts:
            if m.shape[1] < max_cols:
                pad = scipy.sparse.csr_matrix((m.shape[0], max_cols - m.shape[1]), dtype=np.float32)
                m = scipy.sparse.hstack([m, pad], format="csr")
            padded.append(m)
        merged_sparse = scipy.sparse.vstack(padded, format="csr")
        log.info("Merged sparse: %s, nnz=%d", merged_sparse.shape, merged_sparse.nnz)
        scipy.sparse.save_npz(str(out_dir / "sparse.npz"), merged_sparse)
    else:
        log.warning("No sparse matrices found — dense-only output")
        merged_sparse = None

    # Metadata
    with open(out_dir / "metadata.jsonl", "w") as fh:
        for entry in merged_meta:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    config = {
        "total_chunks":        merged_index.ntotal,
        "num_partial_indexes": len(partial_indexes),
        "index_type":          "IndexFlatIP (cosine)",
        "embedding_model":     "BAAI/bge-m3",
        "has_sparse":          merged_sparse is not None,
        "sparse_shape":        list(merged_sparse.shape) if merged_sparse is not None else None,
        "partial_dir":         str(partial_dir),
    }
    with open(out_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    log.info("Done → %s", out_dir)


if __name__ == "__main__":
    main()
