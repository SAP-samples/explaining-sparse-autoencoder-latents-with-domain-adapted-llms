"""
build_token_map.py

Preprocessing script: builds a token-position → (chunk_idx, tok_pos_in_chunk)
mapping for the CrossCoder activation context lookup feature.

Reads the JSONL dataset used to generate cc_acts, tokenizes each chunk **once**
(with offset_mapping), and saves four files alongside the cc_acts meta:

  <base>_token_map.npy          shape (N, 2)  int32   [chunk_idx, tok_pos_in_chunk]
  <base>_chunk_texts.npy        object array of raw chunk strings
  <base>_chunk_domains.npy      shape (n_chunks,) uint8  0=domain_A  1=domain_B
  <base>_token_char_offsets.npy shape (N, 2)  int32   [char_start, char_end] of
                                  each token within its chunk text (from offset_mapping)

The char_offsets file lets the viewer highlight the exact activating token without
any regex matching — just slice text[char_start:char_end].

Usage:
    python build_token_map.py \\
        --meta  /path/to/cc_acts_..._meta.pt \\
        --tokenizer  google/gemma-2-2b
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta",      required=True, help="Path to cc_acts meta.pt")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer ID or local path")
    parser.add_argument("--max-len",   type=int, default=2048)
    args = parser.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        print(f"ERROR: meta file not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    meta = torch.load(str(meta_path), map_location="cpu", weights_only=False)
    dataset_path = meta["dataset"]
    split        = meta.get("split", "test")
    n_tokens_expected = meta["n_tokens"]

    base = meta_path.parent / meta_path.stem.replace("_meta", "")
    out_token_map     = Path(str(base) + "_token_map.npy")
    out_chunk_texts   = Path(str(base) + "_chunk_texts.npy")
    out_chunk_domains = Path(str(base) + "_chunk_domains.npy")
    out_char_offsets  = Path(str(base) + "_token_char_offsets.npy")

    print(f"Dataset:  {dataset_path}")
    print(f"Split:    {split}")
    print(f"Expected tokens: {n_tokens_expected:,}")
    print(f"Output base: {base}")

    print(f"\nLoading tokenizer: {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset ({split} split) ...")
    chunks = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if split != "all" and rec.get("split") != split:
                continue
            text = rec.get("text", "")
            is_b = rec.get("domain_label", rec.get("japanese_label", False))
            chunks.append((text, 1 if is_b else 0))

    print(f"  Chunks: {len(chunks):,}")

    # Single pass: tokenize with offset_mapping, collect everything at once.
    # We accumulate per-chunk results before final allocation so we only call
    # the tokenizer once per chunk (not twice as before).
    print("\nTokenizing chunks (single pass with offset_mapping) ...")
    chunk_texts   = np.empty(len(chunks), dtype=object)
    chunk_domains = np.zeros(len(chunks), dtype=np.uint8)
    per_chunk_lengths = []
    per_chunk_offsets = []  # list of np arrays shape (n_tok, 2)

    for chunk_idx, (text, domain) in enumerate(chunks):
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_len,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = np.array(enc["offset_mapping"], dtype=np.int32)  # (n_tok, 2)
        n = len(offsets)
        per_chunk_lengths.append(n)
        per_chunk_offsets.append(offsets)
        chunk_texts[chunk_idx]   = text
        chunk_domains[chunk_idx] = domain
        if (chunk_idx + 1) % 1000 == 0:
            print(f"  {chunk_idx+1:,}/{len(chunks):,} chunks tokenized")

    total_tokens = sum(per_chunk_lengths)
    print(f"\n  Total tokens: {total_tokens:,}  (expected: {n_tokens_expected:,})")
    if total_tokens != n_tokens_expected:
        print("  WARNING: token count mismatch — map may be misaligned with cc_acts!")

    # Build arrays
    print("\nBuilding token_map and char_offsets arrays ...")
    token_map    = np.zeros((total_tokens, 2), dtype=np.int32)
    char_offsets = np.zeros((total_tokens, 2), dtype=np.int32)

    global_idx = 0
    for chunk_idx, (offsets, n) in enumerate(zip(per_chunk_offsets, per_chunk_lengths)):
        token_map[global_idx : global_idx + n, 0] = chunk_idx
        token_map[global_idx : global_idx + n, 1] = np.arange(n, dtype=np.int32)
        char_offsets[global_idx : global_idx + n]  = offsets
        global_idx += n

    print(f"\nSaving ...")
    np.save(str(out_token_map),    token_map)
    np.save(str(out_chunk_texts),  chunk_texts)
    np.save(str(out_chunk_domains), chunk_domains)
    np.save(str(out_char_offsets), char_offsets)

    print(f"  token_map:          {out_token_map}  shape={token_map.shape}")
    print(f"  chunk_texts:        {out_chunk_texts}  n={len(chunk_texts)}")
    print(f"  chunk_domains:      {out_chunk_domains}  n={len(chunk_domains)}")
    print(f"  token_char_offsets: {out_char_offsets}  shape={char_offsets.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
