"""
generate_dataset.py

Builds a 50/50 interleaved dataset from two HuggingFace datasets or local JSONL files
for CrossCoder model-diffing experiments.

Supports any two text sources — language pairs, domain pairs (medical/general,
code/prose, business/web), etc.

Output JSONL schema (one line per chunk):
{
  "id":           str,   # uuid5 of (source_id + chunk_index)
  "text":         str,   # raw chunk text (~chunk_tokens chars/4)
  "source_id":    str,   # document id from source dataset
  "source_dir":   str,   # label string passed via --label-a / --label-b
  "chunk_idx":    int,   # 0-based index within source doc
  "token_count":  int,   # approx token count (len(text) // 4)
  "domain_label": bool,  # false = domain A, true = domain B
  "split":        str,   # "train" / "validation" / "test"
}

Usage (HF datasets):
    python generate_dataset.py \\
        --dataset-a  HuggingFaceFW/fineweb --subset-a sample-10BT \\
        --dataset-b  HuggingFaceFW/fineweb-2 --subset-b jpn_Jpan \\
        --label-a    fineweb:en --label-b fineweb2:ja \\
        --total-tokens 20_000_000 \\
        --out /path/to/output.jsonl

Usage (local JSONL, must have 'text' field):
    python generate_dataset.py \\
        --dataset-a  /path/to/domain_a.jsonl \\
        --dataset-b  /path/to/domain_b.jsonl \\
        --label-a    domain-a --label-b domain-b \\
        --total-tokens 20_000_000 \\
        --out /path/to/output.jsonl

Defaults:
    --total-tokens       20_000_000
    --chunk-tokens       2048
    --text-field         text
    --seed               42
    --hf-shuffle-buffer  10_000
"""

import argparse
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

LA_TZ = timezone(timedelta(hours=-7))
CHARS_PER_TOKEN = 4
MIN_TOKENS = 500


def chunk_text(text: str, chunk_chars: int) -> list:
    chunks = []
    for i in range(0, len(text) - chunk_chars + 1, chunk_chars):
        chunks.append(text[i : i + chunk_chars])
    return chunks


def assign_split(chunk_id: str) -> str:
    bucket = int(chunk_id.replace("-", ""), 16) % 100
    if bucket < 70:
        return "train"
    elif bucket < 85:
        return "validation"
    else:
        return "test"


def iter_hf(hf_dataset: str, hf_subset: str, text_field: str, seed: int, shuffle_buffer: int):
    from datasets import load_dataset
    for attempt in range(3):
        try:
            ds = load_dataset(hf_dataset, name=hf_subset or None, split="train", streaming=True)
            break
        except Exception as e:
            if attempt == 2:
                raise
            wait = 10 * (attempt + 1)
            print(f"  [retry {attempt+1}/3] load_dataset failed: {e}  (waiting {wait}s)", flush=True)
            time.sleep(wait)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    first_rec = None
    for rec in ds:
        if first_rec is None:
            first_rec = rec
            available = list(rec.keys())
            if text_field not in rec and text_field != "text":
                print(f"  [warn] text_field='{text_field}' not found in dataset. "
                      f"Available columns: {available}", flush=True)
            elif text_field == "text" and "text" not in rec:
                # Common alternative column names
                for alt in ("article", "content", "passage", "body", "document"):
                    if alt in rec:
                        print(f"  [warn] No 'text' column found; using '{alt}' instead. "
                              f"Pass --text-field {alt} to suppress this warning.", flush=True)
                        text_field = alt
                        break
                else:
                    print(f"  [warn] No 'text' column found. Available: {available}", flush=True)
        text = rec.get(text_field) or ""
        source_id = rec.get("id") or rec.get("url") or str(uuid.uuid4())
        if text:
            yield source_id, text


def iter_jsonl(path: str, text_field: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get(text_field) or rec.get("text", "")
            source_id = rec.get("id") or str(uuid.uuid4())
            if text:
                yield source_id, text


def is_local(path: str) -> bool:
    return Path(path).exists() or path.startswith("/") or path.startswith("./")


def collect_chunks(
    domain_name: str,
    dataset_path: str,
    hf_subset: str,
    text_field: str,
    source_dir_label: str,
    is_domain_b: bool,
    target_tokens: int,
    chunk_chars: int,
    seed: int,
    shuffle_buffer: int,
    existing_ids: set,
    out_f,
) -> int:
    target_chars = target_tokens * CHARS_PER_TOKEN
    print(f"\n=== {domain_name} ===", flush=True)

    if is_local(dataset_path):
        source_iter = iter_jsonl(dataset_path, text_field)
    else:
        source_iter = iter_hf(dataset_path, hf_subset, text_field, seed, shuffle_buffer)

    collected_chars = 0
    total_chunks = 0
    skipped_chunks = 0
    docs_seen = 0
    docs_used = 0

    for source_id, text in source_iter:
        if collected_chars >= target_chars:
            break
        docs_seen += 1
        if len(text) // CHARS_PER_TOKEN < MIN_TOKENS:
            continue
        chunks = chunk_text(text, chunk_chars)
        if not chunks:
            continue
        docs_used += 1
        for chunk_idx, chunk in enumerate(chunks):
            if collected_chars >= target_chars:
                break
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_id}_{chunk_idx}"))
            if chunk_id in existing_ids:
                collected_chars += len(chunk)
                skipped_chunks += 1
                continue
            out_f.write(json.dumps({
                "id":           chunk_id,
                "text":         chunk,
                "source_id":    source_id,
                "source_dir":   source_dir_label,
                "chunk_idx":    chunk_idx,
                "token_count":  len(chunk) // CHARS_PER_TOKEN,
                "domain_label": is_domain_b,
                "split":        assign_split(chunk_id),
            }, ensure_ascii=False) + "\n")
            collected_chars += len(chunk)
            total_chunks += 1

        if (total_chunks + skipped_chunks) % 5_000 == 0 and (total_chunks + skipped_chunks) > 0:
            print(
                f"  {total_chunks:,} new | {skipped_chunks:,} skipped"
                f" | {collected_chars // CHARS_PER_TOKEN / 1_000_000:.2f}M tokens",
                flush=True,
            )

    print(f"  Done: {total_chunks:,} new chunks, {skipped_chunks:,} skipped,"
          f" from {docs_used:,} docs ({docs_seen:,} seen)", flush=True)
    return total_chunks + skipped_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Build a 50/50 CrossCoder dataset from two text sources (HF or local JSONL)"
    )
    parser.add_argument("--dataset-a",  required=True,
                        help="HF dataset ID or local JSONL path — Domain A (base/reference)")
    parser.add_argument("--subset-a",   default=None,
                        help="HF subset/config name for dataset-a (e.g. 'sample-10BT')")
    parser.add_argument("--dataset-b",  required=True,
                        help="HF dataset ID or local JSONL path — Domain B (fine-tuned/CPT)")
    parser.add_argument("--subset-b",   default=None,
                        help="HF subset/config name for dataset-b")
    parser.add_argument("--label-a",    default="domain-a",
                        help="source_dir label for domain A (default: domain-a)")
    parser.add_argument("--label-b",    default="domain-b",
                        help="source_dir label for domain B (default: domain-b)")
    parser.add_argument("--text-field", default="text",
                        help="Field name containing text in each record (default: text)")
    parser.add_argument("--total-tokens", type=int, default=20_000_000,
                        help="Total tokens across both sources (split 50/50)")
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--out", default=None,
                        help="Output JSONL path. Auto-generated if omitted.")
    args = parser.parse_args()

    half_tokens = args.total_tokens // 2
    chunk_chars = args.chunk_tokens * CHARS_PER_TOKEN

    if args.out is None:
        date_str = datetime.now(LA_TZ).strftime("%Y%m%d")
        a_slug = Path(args.dataset_a).stem if is_local(args.dataset_a) else args.dataset_a.split("/")[-1]
        b_slug = Path(args.dataset_b).stem if is_local(args.dataset_b) else args.dataset_b.split("/")[-1]
        total_m = args.total_tokens // 1_000_000
        args.out = f"crosscoder_dataset_{a_slug}_vs_{b_slug}_{total_m}M_{date_str}.jsonl"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set = set()
    if out_path.exists():
        print(f"Resuming: scanning existing file for written IDs ...", flush=True)
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"  Found {len(existing_ids):,} existing chunks — skipping these", flush=True)

    print(f"Target: {args.total_tokens:,} tokens total ({half_tokens:,} domain-A + {half_tokens:,} domain-B)", flush=True)
    print(f"Output: {out_path.resolve()}", flush=True)

    with open(out_path, "a") as out_f:
        a_chunks = collect_chunks(
            domain_name=f"Domain A — {args.dataset_a}",
            dataset_path=args.dataset_a,
            hf_subset=args.subset_a,
            text_field=args.text_field,
            source_dir_label=args.label_a,
            is_domain_b=False,
            target_tokens=half_tokens,
            chunk_chars=chunk_chars,
            seed=args.seed,
            shuffle_buffer=args.hf_shuffle_buffer,
            existing_ids=existing_ids,
            out_f=out_f,
        )
        b_chunks = collect_chunks(
            domain_name=f"Domain B — {args.dataset_b}",
            dataset_path=args.dataset_b,
            hf_subset=args.subset_b,
            text_field=args.text_field,
            source_dir_label=args.label_b,
            is_domain_b=True,
            target_tokens=half_tokens,
            chunk_chars=chunk_chars,
            seed=args.seed,
            shuffle_buffer=args.hf_shuffle_buffer,
            existing_ids=existing_ids,
            out_f=out_f,
        )

    if a_chunks == 0:
        print("ERROR: Domain A produced 0 chunks — check --dataset-a path/ID and --text-field.", flush=True)
        import sys; sys.exit(1)
    if b_chunks == 0:
        print("ERROR: Domain B produced 0 chunks — check --dataset-b path/ID and --text-field.", flush=True)
        import sys; sys.exit(1)

    total_tokens = (a_chunks + b_chunks) * args.chunk_tokens
    print(f"\n=== Combined output ===", flush=True)
    print(f"  Domain A chunks: {a_chunks:,}  (~{a_chunks * args.chunk_tokens / 1_000_000:.2f}M tokens)", flush=True)
    print(f"  Domain B chunks: {b_chunks:,}  (~{b_chunks * args.chunk_tokens / 1_000_000:.2f}M tokens)", flush=True)
    print(f"  Total chunks:    {a_chunks + b_chunks:,}  (~{total_tokens / 1_000_000:.2f}M tokens)", flush=True)
    print(f"  Output:          {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
