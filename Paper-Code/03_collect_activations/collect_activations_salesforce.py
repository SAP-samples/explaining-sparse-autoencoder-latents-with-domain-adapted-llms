"""
collect_activations_salesforce.py

Runs a text dataset through a causal LM and saves the residual stream activations
at a specified layer for every token. Adapted for Llama-3-8B architecture.

The residual stream at layer L is defined as the output of model.layers[L] — i.e.
the hidden state after attention + MLP + residual addition at that layer.

A forward hook is registered on model.layers[--layer] to capture activations without
modifying the model. The model itself is never modified and no gradients are computed.

For Llama-3-8B the last layer is 31 (32 layers total, 0-indexed).
d_model = 4096.

Output: one .pt metadata file + one .mmap data file of shape (N, d_model) float32
where N = total number of tokens across all chunks in the dataset split.

Usage:
    python collect_activations_salesforce.py \\
        --model   <path_or_hf_id> \\
        --dataset <path_to_jsonl> \\
        --layer   31 \\
        --split   train \\
        --out     <output_dir>

Defaults:
    --layer        31   (last layer of Llama-3-8B)
    --split        train
    --lang         all  (all | en | salesforce)
    --batch        8    (chunks per forward pass)
    --max-len      2048
    --out          ./activations
"""

import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LA_TZ = timezone(timedelta(hours=-7))


# ---------------------------------------------------------------------------
# Activation collection via forward hook
# ---------------------------------------------------------------------------

def register_hook(model, layer_idx: int):
    """Register a forward hook on the residual stream at layer_idx.
    Supports:
      - Llama-3 / most HF causal LMs: model.model.layers
      - Gemma-3 (language_model wrapper): model.model.language_model.layers
    Returns (handle, store). store['acts'] is set after each forward pass.
    """
    store = {}

    def hook(module, input, output):  # noqa: ARG001
        hidden = output[0] if isinstance(output, tuple) else output
        store["acts"] = hidden.detach().cpu()

    inner = model.model
    # Gemma-3 wraps the text model under language_model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    handle = inner.layers[layer_idx].register_forward_hook(hook)
    return handle, store


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--layer", type=int, default=31,
                        help="Layer index to hook (Llama-3-8B last layer = 31)")
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "test", "all"])
    parser.add_argument("--lang", default="all",
                        choices=["all", "en", "salesforce"],
                        help="Filter by domain: en (salesforce_label=false), "
                             "salesforce (salesforce_label=true), all (no filter)")
    parser.add_argument("--batch", type=int, default=8, help="Chunks per forward pass")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--out", default="./activations",
                        help="Directory for .pt metadata files")
    parser.add_argument("--mmap-dir", default=None,
                        help="Directory for .mmap data files (defaults to --out)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mmap_dir = Path(args.mmap_dir) if args.mmap_dir else out_dir
    mmap_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.model).name if "/" not in args.model else args.model.split("/")[-1]
    dataset_name = Path(args.dataset).stem
    date_str = datetime.now(LA_TZ).strftime("%Y%m%d")
    lang_suffix = f"_{args.lang}" if args.lang != "all" else ""
    stem = f"acts_{model_name}_{dataset_name}_layer{args.layer}_{args.split}{lang_suffix}_{date_str}"
    out_path = out_dir / f"{stem}.pt"
    mmap_path = mmap_dir / f"{stem}.mmap"

    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.dataset}")
    print(f"Layer:      {args.layer}")
    print(f"Split:      {args.split}")
    print(f"Lang:       {args.lang}")
    print(f"Output:     {out_path}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Load tokenizer + model
    # -----------------------------------------------------------------------
    print("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model ...")
    n_gpus = torch.cuda.device_count()
    print(f"  GPUs available: {n_gpus}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded on: {next(model.parameters()).device}")

    # -----------------------------------------------------------------------
    # Register hook
    # -----------------------------------------------------------------------
    hook_handle, act_store = register_hook(model, args.layer)
    print(f"  Hook registered on model.layers[{args.layer}]")

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print(f"\nLoading dataset (split={args.split}, lang={args.lang}) ...")
    texts = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if args.split != "all" and rec.get("split") != args.split:
                continue
            if args.lang == "en" and rec.get("salesforce_label", False):
                continue
            if args.lang == "salesforce" and not rec.get("salesforce_label", False):
                continue
            texts.append(rec["text"])

    print(f"  Chunks loaded: {len(texts):,}")

    # -----------------------------------------------------------------------
    # First pass: count total tokens to pre-allocate mmap
    # -----------------------------------------------------------------------
    print(f"\nCounting tokens for pre-allocation ...")
    total_tokens = 0
    for text in texts:
        ids = tokenizer(text, truncation=True, max_length=args.max_len)["input_ids"]
        total_tokens += len(ids)
    print(f"  Total tokens: {total_tokens:,}")

    # -----------------------------------------------------------------------
    # Collect activations, write directly to pre-allocated mmap
    # -----------------------------------------------------------------------
    d_model_probe = None
    mmap = None
    write_idx = 0

    print(f"\nCollecting activations (batch_size={args.batch}) ...")
    total_batches = (len(texts) + args.batch - 1) // args.batch
    t_start = time.time()

    for batch_start in range(0, len(texts), args.batch):
        batch_texts = texts[batch_start : batch_start + args.batch]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        acts = act_store["acts"]  # (batch, seq_len, d_model), CPU bfloat16

        # Create mmap on first batch once we know d_model
        if mmap is None:
            d_model_probe = acts.shape[-1]
            print(f"  d_model={d_model_probe}, pre-allocating {mmap_path} ...")
            mmap = np.memmap(mmap_path, dtype="float32", mode="w+",
                             shape=(total_tokens, d_model_probe))
            print(f"  Pre-allocated {total_tokens * d_model_probe * 4 / 1e9:.2f} GB")

        mask = inputs["attention_mask"].bool()
        for i in range(acts.shape[0]):
            valid = acts[i][mask[i]].to(torch.float32).numpy()
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

    hook_handle.remove()
    mmap.flush()

    # -----------------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------------
    print(f"\nSaving metadata to {out_path} ...")
    torch.save({
        "mmap_path": str(mmap_path),
        "shape": (write_idx, d_model_probe),
        "dtype": "float32",
        "model": args.model,
        "dataset": args.dataset,
        "layer": args.layer,
        "split": args.split,
        "lang": args.lang,
        "n_tokens": write_idx,
        "d_model": d_model_probe,
    }, out_path)

    size_gb = mmap_path.stat().st_size / 1e9
    print(f"\nDone.")
    print(f"  Tokens:    {write_idx:,}")
    print(f"  Shape:     ({write_idx}, {d_model_probe})")
    print(f"  File size: {size_gb:.2f} GB")
    print(f"  Data:      {mmap_path.resolve()}")
    print(f"  Metadata:  {out_path.resolve()}")


if __name__ == "__main__":
    main()
