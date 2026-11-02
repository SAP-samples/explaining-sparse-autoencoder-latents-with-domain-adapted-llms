"""
collect_crosscoder_acts.py

Runs a paired dataset through two LMs (base + CPT) and a trained CrossCoder,
saving the sparse CrossCoder feature activations for every token.

Architecture:
  - Hook residual stream at --layer from both models simultaneously
  - Separate encoders (v5): x_A @ W_enc_A + x_B @ W_enc_B + b_enc → TopK/BatchTopK
  - Store top-K indices + values per token
  - Also saves domain label per token (domain_label field from dataset)

Output (saved to --out-dir):
  cc_acts_<run_name>_indices.npy   shape (N, K) int32   — active feature indices
  cc_acts_<run_name>_values.npy    shape (N, K) float32 — activation strengths
  cc_acts_<run_name>_domain.npy    shape (N,)   uint8   — 0=domain A, 1=domain B
  cc_acts_<run_name>_meta.pt       metadata dict

Usage:
    python collect_crosscoder_acts.py \\
        --model-base    google/gemma-2-2b \\
        --model-cpt     /path/to/swallow \\
        --crosscoder    /path/to/cc_checkpoint.pt \\
        --dataset       /path/to/dataset.jsonl \\
        --split         test \\
        --layer         25 \\
        --out-dir       ./cc_activations
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# shared utilities: architecture-agnostic layer lookup + inference CrossCoder model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_utils import get_layer_module
from crosscoder_model import CrossCoderV5

LA_TZ = timezone(timedelta(hours=-7))


# ---------------------------------------------------------------------------
# Hook helper
# ---------------------------------------------------------------------------

def register_hook(model):
    store = {}
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Convert to float32 on GPU before CPU transfer to reduce PCIe bandwidth
        store["acts"] = hidden.detach().to(torch.float32).cpu()
    return store, hook


def attach_hook(model, layer_idx: int, store: dict, hook_fn):
    layer = get_layer_module(model, layer_idx)
    handle = layer.register_forward_hook(hook_fn)
    return handle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base",  required=True, help="Base LM (HF name or local path)")
    parser.add_argument("--model-cpt",   required=True, help="CPT LM (HF name or local path)")
    parser.add_argument("--crosscoder",  required=True, help="CrossCoder .pt checkpoint")
    parser.add_argument("--dataset",     required=True, help="JSONL dataset with 'text' and 'split' fields")
    parser.add_argument("--split",       default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--layer",       type=int, default=25)
    parser.add_argument("--batch",       type=int, default=2)
    parser.add_argument("--max-len",     type=int, default=2048)
    parser.add_argument("--out-dir",     default="./cc_activations")
    parser.add_argument("--sequential",  action="store_true",
                        help="Warn if both models together exceed 80%% of available GPU VRAM")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str    = datetime.now(LA_TZ).strftime("%Y%m%d_%H%M%S")
    base_name   = Path(args.model_base).name
    cpt_name    = Path(args.model_cpt).name
    cc_stem     = Path(args.crosscoder).stem
    dataset_name = Path(args.dataset).stem
    run_name    = f"cc_{base_name}_vs_{cpt_name}_{cc_stem}_{dataset_name}_layer{args.layer}_{args.split}_{date_str}"

    indices_path  = out_dir / f"cc_acts_{run_name}_indices.npy"
    values_path   = out_dir / f"cc_acts_{run_name}_values.npy"
    domain_path   = out_dir / f"cc_acts_{run_name}_domain.npy"
    meta_path     = out_dir / f"cc_acts_{run_name}_meta.pt"

    print("=" * 70)
    print(f"Base model:  {args.model_base}")
    print(f"CPT model:   {args.model_cpt}")
    print(f"CrossCoder:  {args.crosscoder}")
    print(f"Dataset:     {args.dataset}  (split={args.split})")
    print(f"Layer:       {args.layer}")
    print(f"Out dir:     {out_dir}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load CrossCoder
    # -----------------------------------------------------------------------
    print("\nLoading CrossCoder ...")
    cc_ckpt = torch.load(args.crosscoder, map_location="cpu", weights_only=False)
    cfg     = cc_ckpt["config"]
    d_model    = cfg["d_model"]
    n_features = cfg["n_features"]
    k          = cfg["topk"]
    arch       = cfg.get("arch", "unknown")
    print(f"  d_model={d_model}  n_features={n_features}  k={k}  arch={arch}")

    cc = CrossCoderV5(d_model, n_features, k)
    cc.load_state_dict(cc_ckpt["model_state_dict"])
    cc.eval()
    cc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cc = cc.to(cc_device)

    # target_rms normalization is baked into the model's encode() via rms_A/rms_B buffers
    target_rms = None

    # Load pre-computed feature classification if available
    feature_classification = cc_ckpt.get("feature_classification", {})
    freq_A = cc_ckpt.get("freq_A", None)
    freq_B = cc_ckpt.get("freq_B", None)
    if feature_classification:
        print(f"  Feature classification: {feature_classification}")

    # -----------------------------------------------------------------------
    # Load tokenizer (shared — both models use Gemma tokenizer)
    # -----------------------------------------------------------------------
    print("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)

    # -----------------------------------------------------------------------
    # Load both LMs
    # -----------------------------------------------------------------------
    print("\nLoading base model ...")
    lm_base = AutoModelForCausalLM.from_pretrained(
        args.model_base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lm_base.eval()

    print("Loading CPT model ...")
    lm_cpt = AutoModelForCausalLM.from_pretrained(
        args.model_cpt, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lm_cpt.eval()

    # Register hooks
    store_base, hook_base = register_hook(lm_base)
    store_cpt,  hook_cpt  = register_hook(lm_cpt)
    h_base = attach_hook(lm_base, args.layer, store_base, hook_base)
    h_cpt  = attach_hook(lm_cpt,  args.layer, store_cpt,  hook_cpt)
    print(f"  Hooks registered on layers[{args.layer}] for both models")

    # Warn if both models together use >80% of available VRAM
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        used_vram  = torch.cuda.memory_reserved(0)
        pct_used   = used_vram / total_vram * 100
        if pct_used > 80:
            print(
                f"\n  WARNING: Both models together occupy {pct_used:.0f}% of GPU VRAM "
                f"({used_vram / 1e9:.1f} GB / {total_vram / 1e9:.1f} GB). "
                f"Consider using a single larger GPU or reducing batch size.",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print(f"\nLoading dataset (split={args.split}) ...")
    texts = []
    domain_labels = []  # 0=domain A, 1=domain B per chunk
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if args.split != "all" and rec.get("split") != args.split:
                continue
            texts.append(rec["text"])
            # accept both domain_label (new) and japanese_label (legacy)
            is_b = rec.get("domain_label", rec.get("japanese_label", False))
            domain_labels.append(1 if is_b else 0)
    n_a = domain_labels.count(0)
    n_b = domain_labels.count(1)
    print(f"  Chunks loaded: {len(texts):,}  (domain_a={n_a}, domain_b={n_b})")

    # -----------------------------------------------------------------------
    # Count tokens for pre-allocation
    # -----------------------------------------------------------------------
    print("\nCounting tokens ...")
    total_tokens = 0
    for text in texts:
        ids = tokenizer(text, truncation=True, max_length=args.max_len)["input_ids"]
        total_tokens += len(ids)
    print(f"  Total tokens: {total_tokens:,}")

    all_indices = np.zeros((total_tokens, k), dtype=np.int32)
    all_values  = np.zeros((total_tokens, k), dtype=np.float32)
    all_domain  = np.zeros(total_tokens, dtype=np.uint8)
    size_gb = total_tokens * k * 8 / 1e9
    print(f"  Pre-allocated {size_gb:.2f} GB")

    # -----------------------------------------------------------------------
    # Collect
    # -----------------------------------------------------------------------
    print(f"\nCollecting CrossCoder activations (batch={args.batch}) ...")
    write_idx    = 0
    total_batches = math.ceil(len(texts) / args.batch)
    t_start = time.time()

    for batch_start in range(0, len(texts), args.batch):
        batch_texts   = texts[batch_start : batch_start + args.batch]
        batch_domains = domain_labels[batch_start : batch_start + args.batch]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
        )
        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        base_device = next(lm_base.parameters()).device
        cpt_device  = next(lm_cpt.parameters()).device

        store_base.clear()  # prevent stale activations if hook doesn't fire
        store_cpt.clear()
        with torch.no_grad():
            lm_base(input_ids=input_ids.to(base_device),
                    attention_mask=attention_mask.to(base_device))
            lm_cpt(input_ids=input_ids.to(cpt_device),
                   attention_mask=attention_mask.to(cpt_device))

        if "acts" not in store_base or "acts" not in store_cpt:
            raise RuntimeError(
                f"Hook did not fire for batch at index {batch_start}. "
                f"Check that layer {args.layer} exists in both models."
            )

        acts_base = store_base["acts"]  # (batch, seq, d_model), cpu float32
        acts_cpt  = store_cpt["acts"]

        mask = inputs["attention_mask"].bool()
        for i in range(acts_base.shape[0]):
            x_A = acts_base[i][mask[i]].to(cc_device)
            x_B = acts_cpt[i][mask[i]].to(cc_device)

            # Apply target_rms normalization if checkpoint has it
            if target_rms is not None:
                x_A = x_A / target_rms.get("scale_A", 1.0)
                x_B = x_B / target_rms.get("scale_B", 1.0)

            with torch.no_grad():
                features = cc.encode(x_A, x_B)  # (n_valid, n_features)

            topk_vals, topk_idx = features.topk(k, dim=-1)
            n = x_A.shape[0]
            all_indices[write_idx : write_idx + n] = topk_idx.cpu().numpy()
            all_values [write_idx : write_idx + n] = topk_vals.cpu().numpy()
            all_domain [write_idx : write_idx + n] = batch_domains[i]
            write_idx += n

        current_batch = batch_start // args.batch + 1
        if current_batch % 20 == 0:
            elapsed = time.time() - t_start
            eta = (total_batches - current_batch) / (current_batch / elapsed)
            print(
                f"  Batch {current_batch:,}/{total_batches:,}"
                f" | tokens: {write_idx:,}"
                f" | elapsed: {str(timedelta(seconds=int(elapsed)))}"
                f" | ETA: {str(timedelta(seconds=int(eta)))}",
                flush=True,
            )

    h_base.remove()
    h_cpt.remove()

    # -----------------------------------------------------------------------
    # Feature firing stats
    # -----------------------------------------------------------------------
    print("\nComputing feature firing stats ...")
    fired = np.zeros(n_features, dtype=bool)
    fired[np.unique(all_indices[:write_idx])] = True
    n_dead  = int((~fired).sum())
    pct_dead = 100.0 * n_dead / n_features
    print(f"  Dead features (never fired): {n_dead:,} / {n_features:,} ({pct_dead:.1f}%)")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print("\nSaving ...")
    np.save(indices_path, all_indices[:write_idx])
    np.save(values_path,  all_values[:write_idx])
    np.save(domain_path,  all_domain[:write_idx])

    meta = {
        "model_base":    args.model_base,
        "model_cpt":     args.model_cpt,
        "crosscoder":    args.crosscoder,
        "dataset":       args.dataset,
        "split":         args.split,
        "layer":         args.layer,
        "n_tokens":      write_idx,
        "n_features":    n_features,
        "topk":          k,
        "d_model":       d_model,
        "arch":          arch,
        "run_name":      run_name,
        "indices_path":  str(indices_path),
        "values_path":   str(values_path),
        "domain_path":   str(domain_path),
        "n_dead_features":   n_dead,
        "pct_dead_features": pct_dead,
        "feature_classification": feature_classification,
    }
    if freq_A is not None:
        meta["freq_A"] = freq_A
    if freq_B is not None:
        meta["freq_B"] = freq_B

    torch.save(meta, meta_path)

    print(f"\nDone.")
    print(f"  Tokens:   {write_idx:,}")
    print(f"  Shape:    ({write_idx}, {k})")
    print(f"  Dead:     {n_dead:,} / {n_features:,} ({pct_dead:.1f}%)")
    print(f"  Indices:  {indices_path}")
    print(f"  Values:   {values_path}")
    print(f"  Domain:   {domain_path}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
