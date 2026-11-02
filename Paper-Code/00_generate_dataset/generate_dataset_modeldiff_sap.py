"""
generate_dataset_modeldiff_sap.py

Builds a 50/50 English-SAP dataset for SAE model-diffing experiments.

Sources:
  fineweb (en)  — English web text from HuggingFaceFW/fineweb (sample-10BT)
  SAP corpus    — /fsx/NOVA/rawri/reconstructed_data_with_description_parsed_text_only/consolidated/consolidated.jsonl

Pipeline:
  1. Stream fineweb English, chunk + collect until --total-tokens // 2
  2. Read SAP JSONL, chunk + collect until --total-tokens // 2 (loops if needed)
  3. Write both to a single JSONL

Output JSONL schema (one line per 2048-token chunk):
{
  "id":         str,   # uuid5 of (source_id + chunk_index)
  "text":       str,   # raw chunk text (~2048 tokens)
  "source_id":  str,   # document id from source
  "source_dir": str,   # "fineweb2:en" or "sap:sap-help-html5-v6"
  "chunk_idx":  int,   # 0-based index within source doc
  "token_count": int,  # approx token count (len(text) // 4)
  "sap_label":  bool,  # true for SAP content, false for English
  "split":      str,   # "train" / "validation" / "test"
}

Usage:
    python generate_dataset_modeldiff_sap.py \\
        --total-tokens 20_000_000 \\
        --out /path/to/output.jsonl

Defaults:
    --total-tokens       20_000_000
    --chunk-tokens       2048
    --seed               42
    --hf-shuffle-buffer  10_000
    --sap-data           /fsx/NOVA/rawri/reconstructed_data_with_description_parsed_text_only/consolidated/consolidated.jsonl
    --out                auto-generated from params + date
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

from datasets import load_dataset

LA_TZ = timezone(timedelta(hours=-7))
CHARS_PER_TOKEN = 4
MIN_TOKENS = 500

SAP_JSONL_DEFAULT = (
    "/fsx/NOVA/rawri/reconstructed_data_with_description_parsed_text_only"
    "/consolidated/consolidated.jsonl"
)


def chunk_text(text: str, chunk_chars: int) -> list[str]:
    """Split text into fixed-size character chunks; trailing short chunk discarded."""
    chunks = []
    for i in range(0, len(text) - chunk_chars + 1, chunk_chars):
        chunks.append(text[i : i + chunk_chars])
    return chunks


def assign_split(chunk_id: str) -> str:
    """Deterministically assign train/validation/test via UUID hash (70/15/15)."""
    bucket = int(chunk_id.replace("-", ""), 16) % 100
    if bucket < 70:
        return "train"
    elif bucket < 85:
        return "validation"
    else:
        return "test"


def collect_chunks_en(target_tokens: int, chunk_chars: int,
                      seed: int, shuffle_buffer: int,
                      existing_ids: set, out_f) -> int:
    """Stream fineweb English, chunk docs, write to out_f until target_tokens reached."""
    target_chars = target_tokens * CHARS_PER_TOKEN

    print(f"\n=== fineweb (en) ===")
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    collected_chars = 0
    total_chunks = 0
    skipped_chunks = 0
    docs_seen = 0
    docs_used = 0

    for rec in ds:
        if collected_chars >= target_chars:
            break
        docs_seen += 1
        text = rec.get("text", "")
        if len(text) // CHARS_PER_TOKEN < MIN_TOKENS:
            continue
        chunks = chunk_text(text, chunk_chars)
        if not chunks:
            continue
        docs_used += 1
        source_id = rec.get("id", str(uuid.uuid4()))
        for chunk_idx, chunk in enumerate(chunks):
            if collected_chars >= target_chars:
                break
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_id}_{chunk_idx}"))
            if chunk_id in existing_ids:
                collected_chars += len(chunk)
                skipped_chunks += 1
                continue
            out_f.write(json.dumps({
                "id":          chunk_id,
                "text":        chunk,
                "source_id":   source_id,
                "source_dir":  "fineweb2:en",
                "chunk_idx":   chunk_idx,
                "token_count": len(chunk) // CHARS_PER_TOKEN,
                "sap_label":   False,
                "split":       assign_split(chunk_id),
            }, ensure_ascii=False) + "\n")
            collected_chars += len(chunk)
            total_chunks += 1

        if (total_chunks + skipped_chunks) % 5_000 == 0 and (total_chunks + skipped_chunks) > 0:
            print(
                f"  {total_chunks:,} new | {skipped_chunks:,} skipped"
                f" | {collected_chars // CHARS_PER_TOKEN / 1_000_000:.1f}M tokens",
                flush=True,
            )

    print(f"  Done: {total_chunks:,} new chunks, {skipped_chunks:,} skipped,"
          f" from {docs_used:,} docs ({docs_seen:,} seen)")
    return total_chunks + skipped_chunks


def collect_chunks_sap(sap_path: str, target_tokens: int, chunk_chars: int,
                       seed: int, existing_ids: set, out_f) -> int:
    """Read SAP JSONL, chunk docs, write to out_f until target_tokens reached.
    Loops over the file if needed (SAP corpus ~1B tokens, so 10M is always one pass)."""
    target_chars = target_tokens * CHARS_PER_TOKEN

    print(f"\n=== SAP corpus ({sap_path}) ===")

    # Load all docs into memory (9k docs, manageable), shuffle for variety
    docs = []
    with open(sap_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                text = rec.get("text", "")
                if len(text) // CHARS_PER_TOKEN >= MIN_TOKENS:
                    docs.append(text)
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(docs):,} SAP docs")
    rng = random.Random(seed)
    rng.shuffle(docs)

    collected_chars = 0
    total_chunks = 0
    skipped_chunks = 0
    docs_used = 0
    pass_num = 0

    while collected_chars < target_chars:
        pass_num += 1
        if pass_num > 1:
            print(f"  Starting pass {pass_num} over SAP corpus ...")
            rng.shuffle(docs)

        for doc_idx, text in enumerate(docs):
            if collected_chars >= target_chars:
                break
            chunks = chunk_text(text, chunk_chars)
            if not chunks:
                continue
            docs_used += 1
            # Stable source_id: hash of pass + doc index
            source_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"sap_pass{pass_num}_doc{doc_idx}"))
            for chunk_idx, chunk in enumerate(chunks):
                if collected_chars >= target_chars:
                    break
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_id}_{chunk_idx}"))
                if chunk_id in existing_ids:
                    collected_chars += len(chunk)
                    skipped_chunks += 1
                    continue
                out_f.write(json.dumps({
                    "id":          chunk_id,
                    "text":        chunk,
                    "source_id":   source_id,
                    "source_dir":  "sap:sap-help-html5-v6",
                    "chunk_idx":   chunk_idx,
                    "token_count": len(chunk) // CHARS_PER_TOKEN,
                    "sap_label":   True,
                    "split":       assign_split(chunk_id),
                }, ensure_ascii=False) + "\n")
                collected_chars += len(chunk)
                total_chunks += 1

            if (total_chunks + skipped_chunks) % 5_000 == 0 and (total_chunks + skipped_chunks) > 0:
                print(
                    f"  {total_chunks:,} new | {skipped_chunks:,} skipped"
                    f" | {collected_chars // CHARS_PER_TOKEN / 1_000_000:.1f}M tokens",
                    flush=True,
                )

    print(f"  Done: {total_chunks:,} new chunks, {skipped_chunks:,} skipped,"
          f" from {docs_used:,} docs ({pass_num} pass(es))")
    return total_chunks + skipped_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tokens", type=int, default=20_000_000,
                        help="Total tokens across both sources (split 50/50)")
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--sap-data", default=SAP_JSONL_DEFAULT,
                        help="Path to SAP consolidated JSONL")
    parser.add_argument("--out", default=None,
                        help="Output path. Auto-generated if omitted.")
    args = parser.parse_args()

    half_tokens = args.total_tokens // 2
    chunk_chars = args.chunk_tokens * CHARS_PER_TOKEN

    if args.out is None:
        date_str = datetime.now(LA_TZ).strftime("%Y%m%d")
        total_m = args.total_tokens // 1_000_000
        args.out = (
            f"sae_dataset_modeldiff_sap_en{total_m // 2}M_sap{total_m // 2}M_{date_str}.jsonl"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: scan existing file for already-written IDs
    existing_ids: set[str] = set()
    if out_path.exists():
        print(f"Resuming: scanning existing file for written IDs ...")
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"  Found {len(existing_ids):,} existing chunks — skipping these")

    print(f"Target: {args.total_tokens:,} tokens total ({half_tokens:,} en + {half_tokens:,} sap)")
    print(f"Output: {out_path.resolve()}")

    with open(out_path, "a") as out_f:
        en_chunks = collect_chunks_en(
            half_tokens, chunk_chars, args.seed, args.hf_shuffle_buffer, existing_ids, out_f
        )
        sap_chunks = collect_chunks_sap(
            args.sap_data, half_tokens, chunk_chars, args.seed, existing_ids, out_f
        )

    total_tokens = (en_chunks + sap_chunks) * args.chunk_tokens
    print(f"\n=== Combined output ===")
    print(f"  English chunks:  {en_chunks:,}  (~{en_chunks * args.chunk_tokens / 1_000_000:.0f}M tokens)")
    print(f"  SAP chunks:      {sap_chunks:,}  (~{sap_chunks * args.chunk_tokens / 1_000_000:.0f}M tokens)")
    print(f"  Total chunks:    {en_chunks + sap_chunks:,}  (~{total_tokens / 1_000_000:.0f}M tokens)")
    print(f"  Output:          {out_path.resolve()}")


if __name__ == "__main__":
    main()
