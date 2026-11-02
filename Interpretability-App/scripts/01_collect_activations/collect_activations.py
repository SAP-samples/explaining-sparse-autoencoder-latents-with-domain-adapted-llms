"""
collect_activations.py

Runs a text dataset through a causal LM and saves residual stream activations
at a specified layer for every token.

Works with any model loadable via AutoModelForCausalLM (Gemma, LLaMA, Mistral, etc.).

The residual stream at layer L is the output of model.model.layers[L] — hidden state
after attention + MLP + residual addition at that layer.

Output: one .pt metadata file + one .mmap data file of shape (N, d_model) float32
where N = total number of tokens across all chunks in the requested split.

Usage:
    python collect_activations.py \\
        --model   google/gemma-2-2b \\
        --dataset /path/to/dataset.jsonl \\
        --layer   25 \\
        --split   train \\
        --domain  all \\
        --out     /path/to/activations

Defaults:
    --layer        25
    --split        train
    --domain       all  (all | a | b)
    --batch        8
    --max-len      2048
    --out          ./activations
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# shared architecture-agnostic layer lookup
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import get_layer_module

LA_TZ = timezone(timedelta(hours=-7))


def register_hook(model, layer_idx: int):
    """Register a forward hook on the residual stream at layer_idx."""
    store = {}

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Convert to float32 on GPU before CPU transfer to reduce PCIe bandwidth
        store["acts"] = hidden.detach().to(torch.float32).cpu()

    layer = get_layer_module(model, layer_idx)
    handle = layer.register_forward_hook(hook)
    return handle, store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="HF model ID or local path")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--layer",   type=int, default=25, help="Layer index to hook")
    parser.add_argument("--split",   default="train",
                        choices=["train", "validation", "test", "all"])
    parser.add_argument("--domain",  default="all",
                        choices=["all", "a", "b"],
                        help="Filter by domain: a (domain_label=false), b (domain_label=true), all (no filter)")
    parser.add_argument("--batch",   type=int, default=2, help="Chunks per forward pass")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--out",     default="./activations",
                        help="Output path: if ends with .pt, used as explicit metadata path; "
                             "otherwise treated as directory and filename is auto-generated")
    parser.add_argument("--mmap-dir", default=None,
                        help="Directory for .mmap data files (defaults to parent of --out)")
    args = parser.parse_args()

    out_arg = Path(args.out)
    if args.out.endswith(".pt"):
        # Explicit file path — use exactly as given
        out_path = out_arg
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mmap_dir = Path(args.mmap_dir) if args.mmap_dir else out_path.parent
        mmap_dir.mkdir(parents=True, exist_ok=True)
        mmap_path = mmap_dir / (out_path.stem + ".mmap")
    else:
        # Directory mode — auto-generate filename
        out_dir = out_arg
        out_dir.mkdir(parents=True, exist_ok=True)
        mmap_dir = Path(args.mmap_dir) if args.mmap_dir else out_dir
        mmap_dir.mkdir(parents=True, exist_ok=True)
        model_name   = Path(args.model).name if "/" not in args.model else args.model.split("/")[-1]
        dataset_name = Path(args.dataset).stem
        date_str     = datetime.now(LA_TZ).strftime("%Y%m%d")
        domain_suffix = f"_{args.domain}" if args.domain != "all" else ""
        stem    = f"acts_{model_name}_{dataset_name}_layer{args.layer}_{args.split}{domain_suffix}_{date_str}"
        out_path  = out_dir / f"{stem}.pt"
        mmap_path = mmap_dir / f"{stem}.mmap"

    print("=" * 60, flush=True)
    print(f"Model:      {args.model}", flush=True)
    print(f"Dataset:    {args.dataset}", flush=True)
    print(f"Layer:      {args.layer}", flush=True)
    print(f"Split:      {args.split}", flush=True)
    print(f"Domain:     {args.domain}", flush=True)
    print(f"Output:     {out_path}", flush=True)
    print("=" * 60, flush=True)

    print("\nLoading tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model ...", flush=True)
    n_gpus = torch.cuda.device_count()
    print(f"  GPUs available: {n_gpus}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded on: {next(model.parameters()).device}", flush=True)

    hook_handle, act_store = register_hook(model, args.layer)
    layer_mod = get_layer_module(model, args.layer)
    print(f"  Hook registered on {type(layer_mod).__name__} at layer {args.layer}", flush=True)

    print(f"\nLoading dataset (split={args.split}, domain={args.domain}) ...", flush=True)
    texts = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if args.split != "all" and rec.get("split") != args.split:
                continue
            # domain_label: False = domain A, True = domain B
            # also accept legacy japanese_label field
            domain_b = rec.get("domain_label", rec.get("japanese_label", False))
            if args.domain == "a" and domain_b:
                continue
            if args.domain == "b" and not domain_b:
                continue
            texts.append(rec["text"])

    print(f"  Chunks loaded: {len(texts):,}", flush=True)

    if not texts:
        print("ERROR: no texts loaded — check --split / --domain filter against dataset.", flush=True)
        sys.exit(1)

    print(f"\nCounting tokens for pre-allocation ...", flush=True)
    total_tokens = 0
    for text in texts:
        ids = tokenizer(text, truncation=True, max_length=args.max_len)["input_ids"]
        total_tokens += len(ids)
    print(f"  Total tokens: {total_tokens:,}", flush=True)

    d_model_probe = None
    mmap = None
    write_idx = 0

    # With device_map="auto" (multi-GPU), model.device raises AttributeError.
    # Use the first parameter's device as the input device for the first layer.
    _input_device = next(model.parameters()).device

    print(f"\nCollecting activations (batch_size={args.batch}) ...", flush=True)
    total_batches = (len(texts) + args.batch - 1) // args.batch
    t_start = time.time()

    try:
        for batch_start in range(0, len(texts), args.batch):
            batch_texts = texts[batch_start : batch_start + args.batch]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_len,
            )
            input_ids      = inputs["input_ids"].to(_input_device)
            attention_mask = inputs["attention_mask"].to(_input_device)

            act_store.clear()  # prevent stale activations if hook doesn't fire
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            if "acts" not in act_store:
                raise RuntimeError(f"Hook did not fire for batch starting at index {batch_start}. "
                                   f"Check that layer {args.layer} exists in the model.")

            acts = act_store["acts"]  # (batch, seq_len, d_model) CPU float32

            if mmap is None:
                d_model_probe = acts.shape[-1]
                print(f"  d_model={d_model_probe}, pre-allocating {mmap_path} ...", flush=True)
                mmap = np.memmap(mmap_path, dtype="float32", mode="w+",
                                 shape=(total_tokens, d_model_probe))
                print(f"  Pre-allocated {total_tokens * d_model_probe * 4 / 1e9:.2f} GB", flush=True)

            mask = inputs["attention_mask"].bool()
            for i in range(acts.shape[0]):
                valid = acts[i][mask[i]].numpy()  # already float32 from hook
                n = valid.shape[0]
                mmap[write_idx : write_idx + n] = valid
                write_idx += n

            current_batch = batch_start // args.batch + 1
            if current_batch % 50 == 0:
                elapsed = time.time() - t_start
                batches_per_sec = current_batch / elapsed
                remaining_batches = total_batches - current_batch
                eta_sec = remaining_batches / batches_per_sec if batches_per_sec > 0 else 0
                print(
                    f"  Batch {current_batch:,} / {total_batches:,}"
                    f" | tokens so far: {write_idx:,}"
                    f" | elapsed: {str(timedelta(seconds=int(elapsed)))}"
                    f" | ETA: {str(timedelta(seconds=int(eta_sec)))}",
                    flush=True,
                )
    finally:
        hook_handle.remove()
        if mmap is not None:
            mmap.flush()


    # Truncate the mmap file to actual size — pre-allocation may have over-estimated
    # because per-sample truncation lengths vary slightly across the batch.
    if write_idx < total_tokens:
        del mmap  # release memmap before resizing
        actual_bytes = write_idx * d_model_probe * 4  # float32 = 4 bytes
        with open(mmap_path, "r+b") as f:
            f.truncate(actual_bytes)
        print(f"  Trimmed mmap: {total_tokens:,} → {write_idx:,} tokens", flush=True)

    print(f"\nSaving metadata to {out_path} ...", flush=True)
    torch.save({
        "mmap_path":  str(mmap_path),
        "shape":      (write_idx, d_model_probe),
        "dtype":      "float32",
        "model":      args.model,
        "dataset":    args.dataset,
        "layer":      args.layer,
        "split":      args.split,
        "domain":     args.domain,
        "n_tokens":   write_idx,
        "d_model":    d_model_probe,
    }, out_path)

    size_gb = mmap_path.stat().st_size / 1e9
    print(f"\nDone.", flush=True)
    print(f"  Tokens:    {write_idx:,}", flush=True)
    print(f"  Shape:     ({write_idx}, {d_model_probe})", flush=True)
    print(f"  File size: {size_gb:.2f} GB", flush=True)
    print(f"  Data:      {mmap_path.resolve()}", flush=True)
    print(f"  Metadata:  {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
