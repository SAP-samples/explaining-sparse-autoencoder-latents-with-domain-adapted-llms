"""
collect_sae_activations.py

Runs a text dataset through a causal LM + trained SAE and saves the sparse
feature activations for every token. Output is the shared activation matrix
used for all downstream analyses (Feature Search, Specificity, Sensitivity,
Universality, Attribution, LLM-as-a-Judge).

For each token, only the TopK active features are stored (all others are 0
by construction of the TopK SAE). Output is saved in sparse format:
  - indices: (N, K) int32  — which features are active per token
  - values:  (N, K) float32 — activation strength per active feature

Output files (saved to --out-dir):
  sae_acts_<run_name>_indices.npy   — shape (N, K), int32
  sae_acts_<run_name>_values.npy    — shape (N, K), float32
  sae_acts_<run_name>_meta.pt       — metadata (model, sae, dataset, layer, split, n_tokens, K)

Usage:
    python collect_sae_activations.py \\
        --model    <path_to_lm_checkpoint> \\
        --sae      <path_to_sae_.pt> \\
        --dataset  <path_to_jsonl> \\
        --split    test \\
        --layer    12 \\
        --out-dir  ./sae_activations

Defaults:
    --layer    12
    --split    test
    --batch    8
    --max-len  2048
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import math

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

LA_TZ = timezone(timedelta(hours=-7))


# ---------------------------------------------------------------------------
# JumpReLU SAE (matches train_sae_gemma3.py exactly)
# ---------------------------------------------------------------------------

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.log_theta = nn.Parameter(torch.zeros(n_features))

    @property
    def theta(self) -> torch.Tensor:
        return torch.exp(self.log_theta)

    def encode(self, x: torch.Tensor):
        z = self.encoder(x - self.pre_bias)
        acts = z * (z >= self.theta).float()
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts) + self.pre_bias

    def forward(self, x: torch.Tensor):
        return self.encode(x)


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

def register_hook(model, layer_idx: int):
    store = {}

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        store["acts"] = hidden.detach().cpu()

    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    handle = inner.layers[layer_idx].register_forward_hook(hook)
    return handle, store


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="Path to LM checkpoint")
    parser.add_argument("--sae",     required=True, help="Path to SAE .pt checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--layer",   type=int, default=12)
    parser.add_argument("--split",   default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--out-dir", default="./sae_activations")
    parser.add_argument("--var-explained", action="store_true",
                        help="Also compute test MSE loss and variance explained (requires decoder pass)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(LA_TZ).strftime("%Y%m%d_%H%M%S")
    model_name   = Path(args.model).name
    dataset_name = Path(args.dataset).stem
    sae_stem     = Path(args.sae).stem  # includes feat/topk info from run_name
    run_name = f"{model_name}_{sae_stem}_{dataset_name}_layer{args.layer}_{args.split}_{date_str}"

    indices_path = out_dir / f"sae_acts_{run_name}_indices.npy"
    values_path  = out_dir / f"sae_acts_{run_name}_values.npy"
    meta_path    = out_dir / f"sae_acts_{run_name}_meta.pt"

    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"SAE:      {args.sae}")
    print(f"Dataset:  {args.dataset}")
    print(f"Layer:    {args.layer}  Split: {args.split}")
    print(f"Out dir:  {out_dir}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Load SAE
    # -----------------------------------------------------------------------
    print("\nLoading SAE ...")
    sae_ckpt = torch.load(args.sae, map_location="cpu", weights_only=True)
    sae = JumpReLUSAE(
        d_model=sae_ckpt["d_model"],
        n_features=sae_ckpt["n_features"],
    )
    sae.load_state_dict(sae_ckpt["model_state_dict"])
    sae.eval()
    sae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae = sae.to(sae_device)
    norm_scalar = sae_ckpt.get("norm_scalar", 1.0)
    K = int(round(sae_ckpt.get("l0_target", sae_ckpt.get("topk", 150))))
    print(f"  d_model={sae_ckpt['d_model']}  n_features={sae_ckpt['n_features']}  l0_target={K}  norm_scalar={norm_scalar:.2f}")

    # -----------------------------------------------------------------------
    # Load LM
    # -----------------------------------------------------------------------
    print("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading model ...")
    lm = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    lm.eval()
    print(f"  LM loaded on: {next(lm.parameters()).device}")

    hook_handle, act_store = register_hook(lm, args.layer)
    print(f"  Hook registered on model.layers[{args.layer}]")

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print(f"\nLoading dataset (split={args.split}) ...")
    texts = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if args.split != "all" and rec.get("split") != args.split:
                continue
            texts.append(rec["text"])
    print(f"  Chunks loaded: {len(texts):,}")

    # -----------------------------------------------------------------------
    # Count tokens for pre-allocation
    # -----------------------------------------------------------------------
    print("\nCounting tokens for pre-allocation ...")
    total_tokens = 0
    for text in texts:
        ids = tokenizer(text, truncation=True, max_length=args.max_len)["input_ids"]
        total_tokens += len(ids)
    print(f"  Total tokens: {total_tokens:,}")

    # Pre-allocate output arrays
    all_indices = np.zeros((total_tokens, K), dtype=np.int32)
    all_values  = np.zeros((total_tokens, K), dtype=np.float32)
    size_gb = total_tokens * K * (4 + 4) / 1e9
    print(f"  Pre-allocated {size_gb:.2f} GB (indices + values)")

    # -----------------------------------------------------------------------
    # Collect
    # -----------------------------------------------------------------------
    print(f"\nCollecting SAE activations (batch_size={args.batch}) ...")
    write_idx    = 0
    total_batches = (len(texts) + args.batch - 1) // args.batch
    t_start = time.time()

    # For var_explained
    mse_sum      = 0.0
    res_var_sum  = 0.0
    total_var    = 0.0
    n_ve_tokens  = 0

    for batch_start in range(0, len(texts), args.batch):
        batch_texts = texts[batch_start : batch_start + args.batch]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
        )
        input_ids     = inputs["input_ids"].to(next(lm.parameters()).device)
        attention_mask = inputs["attention_mask"].to(next(lm.parameters()).device)

        with torch.no_grad():
            lm(input_ids=input_ids, attention_mask=attention_mask)

        # LM residual stream activations: (batch, seq_len, d_model)
        lm_acts = act_store["acts"]  # CPU, bfloat16

        # Pass through SAE — process valid (non-padding) tokens per example
        mask = inputs["attention_mask"].bool()  # (batch, seq_len)
        for i in range(lm_acts.shape[0]):
            valid_acts = lm_acts[i][mask[i]].to(torch.float32).to(sae_device)  # (n_valid, d_model)
            valid_acts_norm = valid_acts / norm_scalar

            with torch.no_grad():
                acts = sae.encode(valid_acts_norm)  # (n_valid, n_features)

                if args.var_explained:
                    x_hat = sae.decode(acts)   # (n_valid, d_model)
                    residual = valid_acts_norm - x_hat
                    mse_sum     += torch.nn.functional.mse_loss(x_hat, valid_acts_norm).item() * valid_acts_norm.shape[0]
                    res_var_sum += residual.var(dim=0).sum().item()
                    total_var   += valid_acts_norm.var(dim=0).sum().item()
                    n_ve_tokens += valid_acts_norm.shape[0]

            # Extract top-K indices and values (use l0_target as K for storage)
            topk_vals, topk_idx = acts.topk(K, dim=-1)  # both (n_valid, K)

            n = valid_acts.shape[0]
            all_indices[write_idx : write_idx + n] = topk_idx.cpu().numpy()
            all_values [write_idx : write_idx + n] = topk_vals.cpu().numpy()
            write_idx += n

        current_batch = batch_start // args.batch + 1
        if current_batch % 50 == 0:
            elapsed = time.time() - t_start
            batches_per_sec = current_batch / elapsed
            eta_sec = (total_batches - current_batch) / batches_per_sec if batches_per_sec > 0 else 0
            print(
                f"  Batch {current_batch:,} / {total_batches:,}"
                f" | tokens: {write_idx:,}"
                f" | elapsed: {str(timedelta(seconds=int(elapsed)))}"
                f" | ETA: {str(timedelta(seconds=int(eta_sec)))}",
                flush=True,
            )

    hook_handle.remove()

    # -----------------------------------------------------------------------
    # Dead feature count (features that never fired on any token)
    # -----------------------------------------------------------------------
    print(f"\nCounting dead features ...")
    n_features = sae_ckpt["n_features"]
    fired = np.zeros(n_features, dtype=bool)
    unique_feats = np.unique(all_indices[:write_idx])
    fired[unique_feats] = True
    n_dead  = int((~fired).sum())
    n_alive = int(fired.sum())
    pct_dead = 100.0 * n_dead / n_features
    print(f"  Dead features:  {n_dead:,} / {n_features:,}  ({pct_dead:.1f}%)")
    print(f"  Alive features: {n_alive:,} / {n_features:,}  ({100-pct_dead:.1f}%)")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print(f"\nSaving ...")
    np.save(indices_path, all_indices[:write_idx])
    np.save(values_path,  all_values [:write_idx])

    meta = {
        "model":      args.model,
        "sae":        args.sae,
        "sae_stem":   sae_stem,
        "dataset":    args.dataset,
        "layer":      args.layer,
        "split":      args.split,
        "n_tokens":   write_idx,
        "n_features": n_features,
        "topk":       K,
        "d_model":    sae_ckpt["d_model"],
        "indices_path": str(indices_path),
        "values_path":  str(values_path),
        "run_name":     run_name,
        "n_dead_features":  n_dead,
        "n_alive_features": n_alive,
        "pct_dead_features": pct_dead,
    }

    if args.var_explained and n_ve_tokens > 0:
        avg_mse      = mse_sum / n_ve_tokens
        var_exp      = 1.0 - (res_var_sum / n_ve_tokens) / (total_var / n_ve_tokens)
        meta["test_mse"]             = avg_mse
        meta["test_var_explained"]   = var_exp
        meta["test_n_tokens"]        = n_ve_tokens

        ve_path = out_dir / f"sae_acts_{run_name}_var_explained.txt"
        with open(ve_path, "w") as f:
            f.write(f"model:              {args.model}\n")
            f.write(f"sae:                {args.sae}\n")
            f.write(f"split:              {args.split}\n")
            f.write(f"n_tokens:           {n_ve_tokens:,}\n")
            f.write(f"test_mse:           {avg_mse:.6f}\n")
            f.write(f"test_var_explained: {var_exp:.6f}\n")
            f.write(f"n_features:         {n_features:,}\n")
            f.write(f"n_dead_features:    {n_dead:,}  ({pct_dead:.1f}%)\n")
            f.write(f"n_alive_features:   {n_alive:,}  ({100-pct_dead:.1f}%)\n")
        print(f"\n  Test MSE:           {avg_mse:.6f}")
        print(f"  Test var_explained: {var_exp:.6f}")
        print(f"  Saved to: {ve_path}")

    torch.save(meta, meta_path)

    print(f"\nDone.")
    print(f"  Tokens:        {write_idx:,}")
    print(f"  Shape:         ({write_idx}, {K})")
    print(f"  Dead features: {n_dead:,} / {n_features:,}  ({pct_dead:.1f}%)")
    print(f"  Indices:       {indices_path}")
    print(f"  Values:        {values_path}")
    print(f"  Metadata:      {meta_path}")


if __name__ == "__main__":
    main()
