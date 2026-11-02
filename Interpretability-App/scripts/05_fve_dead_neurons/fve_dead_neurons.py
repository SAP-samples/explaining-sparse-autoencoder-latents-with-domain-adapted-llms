"""
fve_dead_neurons.py  —  FVE + Dead Neuron Metrics for CrossCoder v5

Computes on the validation split:
  - FVE_A, FVE_B  : Fraction of Variance Explained per model
                    FVE = 1 - E[‖x - x_hat‖²] / E[‖x‖²]
  - dead_frac     : fraction of features that never activate across the split
  - n_dead        : absolute count of dead features
  - mean_l0       : average number of active features per token (should ≈ topk)
  - total_tokens  : number of tokens evaluated

Results written as JSON to --out.

Usage:
    python fve_dead_neurons.py \
        --model-a   google/gemma-2-2b \
        --model-b   /fsx/.../Gemma-2-Swallow \
        --crosscoder /path/to/cc.pt \
        --dataset   /path/to/dataset.jsonl \
        --layer     25 \
        --split     validation \
        --out       /path/to/fve.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from model_utils import get_layer_module
from crosscoder_model import CrossCoderV5

LA_TZ = timezone(timedelta(hours=-7))


# ── hook helpers ─────────────────────────────────────────────────────────────

def register_hook(model, layer_idx: int):
    store = {}

    def hook(module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        store["acts"] = hidden.detach().cpu()

    layer = get_layer_module(model, layer_idx)
    handle = layer.register_forward_hook(hook)
    return handle, store


# ── load CrossCoder ────────────────────────────────────────────────────────────

def load_crosscoder(path: str, device: torch.device):
    """Load CrossCoder v5 checkpoint. Returns (CrossCoderV5, topk)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg   = ckpt["config"]
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    elif isinstance(ckpt, dict) and any(k.startswith("W_enc") for k in ckpt):
        state = ckpt
        cfg   = {}
    else:
        raise ValueError(f"Unrecognised checkpoint format at {path}")

    n_features = state["W_enc_A"].shape[1]
    d_model    = state["W_enc_A"].shape[0]
    k          = int(cfg.get("topk", 100))

    cc = CrossCoderV5(d_model, n_features, k)
    missing, unexpected = cc.load_state_dict(state, strict=False)
    unexpected_real = [x for x in unexpected if x != "threshold"]
    if unexpected_real:
        print(f"  [warn] unexpected checkpoint keys: {unexpected_real}", file=sys.stderr)
    cc.to(device).eval()
    print(f"  CrossCoder: n_features={n_features}, d_model={d_model}, k={k}")
    return cc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a",    required=True, help="HF ID or path — base model")
    parser.add_argument("--model-b",    required=True, help="HF ID or path — CPT model")
    parser.add_argument("--crosscoder", required=True, help="CrossCoder .pt checkpoint")
    parser.add_argument("--dataset",    required=True, help="JSONL dataset path")
    parser.add_argument("--layer",      type=int, default=25)
    parser.add_argument("--split",      default="validation",
                        choices=["train", "validation", "test", "all"])
    parser.add_argument("--batch",      type=int, default=4)
    parser.add_argument("--max-len",    type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=500_000,
                        help="Max tokens to evaluate (0 = all)")
    parser.add_argument("--out",        required=True, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load CrossCoder
    print(f"Loading CrossCoder: {args.crosscoder}", flush=True)
    cc = load_crosscoder(args.crosscoder, device)

    # Load dataset
    print(f"Loading dataset: {args.dataset}  split={args.split}", flush=True)
    chunks = []
    with open(args.dataset) as f:
        for line in f:
            row = json.loads(line)
            if args.split != "all" and row.get("split") != args.split:
                continue
            chunks.append(row["text"])
    print(f"  {len(chunks)} chunks in split '{args.split}'", flush=True)

    # Load models
    print(f"Loading Model A: {args.model_a}", flush=True)
    tok_a = AutoTokenizer.from_pretrained(args.model_a)
    mdl_a = AutoModelForCausalLM.from_pretrained(
        args.model_a, torch_dtype=torch.float32, device_map="cpu"
    ).eval()
    handle_a, store_a = register_hook(mdl_a, args.layer)
    mdl_a.to(device)

    print(f"Loading Model B: {args.model_b}", flush=True)
    tok_b = AutoTokenizer.from_pretrained(args.model_b)
    mdl_b = AutoModelForCausalLM.from_pretrained(
        args.model_b, torch_dtype=torch.float32, device_map="cpu"
    ).eval()
    handle_b, store_b = register_hook(mdl_b, args.layer)
    mdl_b.to(device)

    # Accumulators
    sum_sq_A = 0.0;  sum_sq_B = 0.0
    sum_err_A = 0.0; sum_err_B = 0.0
    total_tokens = 0
    ever_active = torch.zeros(cc.n_features, dtype=torch.bool)
    l0_sum = 0.0

    max_tok = args.max_tokens if args.max_tokens > 0 else float("inf")

    with torch.no_grad():
        for i in range(0, len(chunks), args.batch):
            if total_tokens >= max_tok:
                break
            batch_texts = chunks[i : i + args.batch]

            enc_a = tok_a(batch_texts, return_tensors="pt", truncation=True,
                          max_length=args.max_len, padding=True).to(device)
            enc_b = tok_b(batch_texts, return_tensors="pt", truncation=True,
                          max_length=args.max_len, padding=True).to(device)

            # Forward through Model A
            _ = mdl_a(**enc_a)
            acts_a = store_a["acts"]  # (B, T, d)

            # Forward through Model B
            _ = mdl_b(**enc_b)
            acts_b = store_b["acts"]  # (B, T, d)

            # Flatten: only non-padding tokens
            mask = enc_a["attention_mask"].bool().cpu()  # (B, T)
            B, T, d = acts_a.shape
            flat_a = acts_a[mask].to(device)  # (N, d)
            flat_b = acts_b[mask[:, :acts_b.shape[1]]].to(device)

            # Align length (tokenizers may differ by 1-2 tokens)
            n = min(flat_a.shape[0], flat_b.shape[0])
            flat_a, flat_b = flat_a[:n], flat_b[:n]

            if n == 0:
                continue

            # CrossCoder forward
            f, x_hat_a, x_hat_b = cc(flat_a, flat_b)

            # Accumulate variance / error
            sum_sq_A  += flat_a.pow(2).sum().item()
            sum_sq_B  += flat_b.pow(2).sum().item()
            sum_err_A += (flat_a - x_hat_a).pow(2).sum().item()
            sum_err_B += (flat_b - x_hat_b).pow(2).sum().item()
            total_tokens += n

            # Dead neuron tracking
            ever_active |= (f > 0).any(dim=0).cpu()

            # L0
            l0_sum += (f > 0).float().sum(dim=-1).mean().item() * n

            if (i // args.batch) % 20 == 0:
                print(f"  {total_tokens:,} tokens processed...", flush=True)

    handle_a.remove()
    handle_b.remove()

    fve_a = 1.0 - sum_err_A / (sum_sq_A + 1e-8)
    fve_b = 1.0 - sum_err_B / (sum_sq_B + 1e-8)
    n_dead = int((~ever_active).sum().item())
    dead_frac = n_dead / cc.n_features
    mean_l0 = l0_sum / total_tokens if total_tokens > 0 else 0.0

    metrics = {
        "run_id":        Path(args.out).stem,
        "computed_at":   datetime.now(LA_TZ).isoformat(),
        "split":         args.split,
        "total_tokens":  total_tokens,
        "n_features":    cc.n_features,
        "topk":          cc.k,
        "fve_a":         round(fve_a, 6),
        "fve_b":         round(fve_b, 6),
        "fve_mean":      round((fve_a + fve_b) / 2, 6),
        "n_dead":        n_dead,
        "dead_frac":     round(dead_frac, 6),
        "mean_l0":       round(mean_l0, 4),
        "model_a":       args.model_a,
        "model_b":       args.model_b,
        "layer":         args.layer,
        "crosscoder":    args.crosscoder,
    }

    print("\n── FVE + Dead Neuron Results ──────────────────────────────")
    print(f"  FVE Model A:     {fve_a:.4f}  ({fve_a*100:.1f}%)")
    print(f"  FVE Model B:     {fve_b:.4f}  ({fve_b*100:.1f}%)")
    print(f"  FVE Mean:        {(fve_a+fve_b)/2:.4f}")
    print(f"  Dead neurons:    {n_dead:,} / {cc.n_features:,}  ({dead_frac*100:.1f}%)")
    print(f"  Mean L0:         {mean_l0:.1f}  (target k={cc.k})")
    print(f"  Total tokens:    {total_tokens:,}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
