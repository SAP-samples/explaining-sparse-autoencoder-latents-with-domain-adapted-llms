"""
Stage 1 of FinTrain vector DB pipeline.

Streams Salesforce/FinTrain finance_unsup, splits text into overlapping chunks
identical to SAP config (chunk_size=512 tokens, overlap=64 tokens),
and writes chunks evenly into N shard files.

Output: /fsx/i777599/cache/fintrain_shards/shard_{i:04d}.jsonl
Each line: {"source": "salesforce:finance_unsup", "line": <doc_idx>, "text": "...chunk..."}

Usage:
    python create_fintrain_chunks.py --num_shards 30
    python create_fintrain_chunks.py --num_shards 5 --max_docs 5000  # test
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def char_chunks(text, chunk_chars, overlap_chars):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_chars])
        start += chunk_chars - overlap_chars
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards",    type=int, default=30)
    parser.add_argument("--chunk_size",    type=int, default=512,
                        help="Chunk size in tokens (1 token ≈ 4 chars)")
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--output_dir",    type=str,
                        default="/fsx/i777599/cache/fintrain_shards")
    parser.add_argument("--max_docs",      type=int, default=None,
                        help="Cap total docs (for testing)")
    args = parser.parse_args()

    chunk_chars   = args.chunk_size   * 4
    overlap_chars = args.chunk_overlap * 4

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "num_shards":           args.num_shards,
        "chunk_size_tokens":    args.chunk_size,
        "chunk_overlap_tokens": args.chunk_overlap,
        "source":               "Salesforce/FinTrain:finance_unsup",
    }
    with open(out_dir / "shards_config.json", "w") as f:
        json.dump(config, f, indent=2)

    shard_handles = [
        open(out_dir / f"shard_{i:04d}.jsonl", "w") for i in range(args.num_shards)
    ]

    total_chunks = 0
    total_docs   = 0

    try:
        log.info("Streaming Salesforce/FinTrain finance_unsup...")
        ds = load_dataset("Salesforce/FinTrain", "finance_unsup", split="train", streaming=True)

        for doc_idx, example in enumerate(ds):
            text = example.get("text", "").strip()
            if not text:
                continue

            for chunk in char_chunks(text, chunk_chars, overlap_chars):
                shard_idx = total_chunks % args.num_shards
                entry = {
                    "source": "salesforce:finance_unsup",
                    "line":   doc_idx,
                    "text":   chunk,
                }
                shard_handles[shard_idx].write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_chunks += 1

            total_docs += 1
            if total_docs % 100_000 == 0:
                log.info("Docs: %d, chunks: %d", total_docs, total_chunks)

            if args.max_docs and total_docs >= args.max_docs:
                log.info("Reached --max_docs=%d, stopping.", args.max_docs)
                break

    finally:
        for fh in shard_handles:
            fh.close()

    config["total_chunks"] = total_chunks
    config["total_docs"]   = total_docs
    with open(out_dir / "shards_config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("Done. Total docs: %d, total chunks: %d", total_docs, total_chunks)
    log.info("Shards written to: %s", out_dir)

    for i in range(args.num_shards):
        shard_path = out_dir / f"shard_{i:04d}.jsonl"
        size_mb = shard_path.stat().st_size / 1e6
        log.info("  shard_%04d.jsonl : %.1f MB", i, size_mb)


if __name__ == "__main__":
    main()
