"""
crosscoder_llm_judge.py

LLM-as-a-Judge pipeline for CrossCoder features.
Works with any two models (base vs CPT, domain A vs domain B, etc.).

For each alive feature:
  1. Extracts top tokens per activation-strength quantile (10 quantiles, 50 tok/bin)
  2. Sends to LLM judge with generic domain-aware prompt (Domain A vs Domain B)
  3. Judge scores on 3 axes (0-4 each) → total 0-12:
       - domain_purity:     how exclusively does it fire on one domain?
       - content_richness:  meaningful content vs structural noise?
       - specificity:       clear interpretable pattern?
  4. Judge assigns domain_label: "A" | "B" | "BOTH"

Additionally saves per-feature:
  - norm_a:      ||W_dec_A[i]||   (decoder norm for model A)
  - norm_b:      ||W_dec_B[i]||
  - delta_norm:  Δnorm = 0.5*(1 + (‖B‖-‖A‖)/max(‖A‖,‖B‖))  range [0,1]
  - norm_flag:   "A" | "B" | "BOTH"  based on delta_norm threshold
  - freq_a:      activation frequency when encode(x_a, zeros)
  - freq_b:      activation frequency when encode(zeros, x_b)
  - freq_flag:   "A" | "B" | "BOTH"  based on frequency ratio

Output:
  crosscoder_features_<date>.csv  — one row per alive feature

CSV columns:
  feature_idx, domain_label, judge_label, judge_explanation,
  judge_score, judge_domain_purity, judge_content_richness, judge_specificity,
  norm_a, norm_b, delta_norm, norm_flag,
  freq_a, freq_b, freq_flag,
  n_active, mean_activation, max_activation,
  quantile_0_10, quantile_10_20, ..., quantile_90_100

Usage:
    python crosscoder_llm_judge.py \\
        --cc-acts-meta   /path/to/cc_acts_*_meta.pt \\
        --crosscoder     /path/to/cc_checkpoint.pt \\
        --dataset        /path/to/dataset.jsonl \\
        --model-a        google/gemma-2-2b \\
        --out-dir        ./llmaaj_results

Defaults:
    --judge-top-n           999999   (judge all alive features)
    --n-quantiles           10
    --tokens-per-quantile   50
    --context-window        10
    --workers               8
    --judge-model           anthropic--claude-4.6-sonnet
    --delta-norm-threshold  0.1      (|delta_norm| threshold for A/B vs BOTH)
    --freq-ratio-threshold  5.0      (freq_A/freq_B > t → A, < 1/t → B)
    --domain-a-label        "Domain A"   (shown in judge prompt)
    --domain-b-label        "Domain B"
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer
from tqdm import tqdm

# shared utilities: inference CrossCoder model definition
sys.path.insert(0, str(Path(__file__).parent.parent))
from crosscoder_model import CrossCoderV5

load_dotenv()

LA_TZ = timezone(timedelta(hours=-7))


# ---------------------------------------------------------------------------
# LLM Judge prompts — generic, parameterized by domain labels at runtime
# ---------------------------------------------------------------------------

def build_judge_system(domain_a_label: str, domain_b_label: str) -> str:
    return f"""\
You are an expert in neural network interpretability analyzing CrossCoder features \
from two models: Model A ({domain_a_label}) and Model B ({domain_b_label}).
You always respond with valid JSON only — no prose, no markdown, no explanation outside the JSON object.

You will be shown activating tokens for a single CrossCoder feature, organized into \
activation-strength quantiles (0-10% = the strongest activations, 90-100% = weakest). \
Each token is labeled [A] ({domain_a_label}) or [B] ({domain_b_label}), with its activation \
strength and surrounding context.

Your task is to characterize what this CrossCoder feature represents and whether it \
captures something specific to one model/domain or shared between both.

Score the feature on these three criteria (each 0-4):

  domain_purity (0-4):
    How exclusively does this feature fire on tokens from one domain?
    0 = fires roughly equally on both domains — no clear preference.
    1 = fires on both but noticeably more on one domain.
    2 = fires mostly on one domain with occasional tokens from the other.
    3 = fires almost exclusively on one domain, very rare exceptions.
    4 = fires ONLY on one domain. Zero or near-zero cross-domain activation visible.

  content_richness (0-4):
    Does this feature capture meaningful content rather than structural/surface tokens?
    0 = purely structural noise — BOS, EOS, padding, whitespace, newlines only.
    1 = almost entirely punctuation or single characters with no semantic content.
    2 = mostly structural with some content words mixed in.
    3 = mostly meaningful content — recognizable words, topics, or concepts dominate.
    4 = rich, specific semantic content — a clear topic cluster or linguistic/domain concept.

  specificity (0-4):
    Can you form a clear, specific interpretation of what this feature detects?
    0 = no clear pattern, random mix of unrelated tokens with no obvious theme.
    1 = vague pattern — something domain-related but nothing more specific.
    2 = clear pattern but broad (e.g. "technical vocabulary" or "formal register").
    3 = precise, specific pattern (e.g. "medical dosage instructions" or "Japanese topic markers").
    4 = highly precise pattern with unambiguous context and strong semantic coherence.

  judge_score = domain_purity + content_richness + specificity  (range: 0 to 12)

Also classify the domain of this feature:
  domain_label:
    "A"    = fires only or almost only on {domain_a_label} tokens
    "B"    = fires only or almost only on {domain_b_label} tokens
    "BOTH" = fires meaningfully on both domains

Output ONLY this JSON object, nothing else:
{{
  "label": "<short label max 8 words describing what this feature detects>",
  "explanation": "<2-3 sentences: what does this feature detect, in what context, why is it interesting>",
  "domain_label": "<A|B|BOTH>",
  "domain_purity": <0-4>,
  "content_richness": <0-4>,
  "specificity": <0-4>,
  "judge_score": <integer from 0 to 12>
}}
"""


def build_judge_user(domain_a_label: str, domain_b_label: str) -> str:
    return f"""\
Activating tokens organized by activation-strength quantile (0-10% = strongest, 90-100% = weakest).
[A] = {domain_a_label} token, [B] = {domain_b_label} token:

{{quantile_blocks}}

Score this feature and return only valid JSON.
"""


def _format_quantile_blocks(quantile_tokens: dict) -> str:
    parts = []
    for label, toks in quantile_tokens.items():
        parts.append(f"--- Quantile {label} ({len(toks)} tokens) ---")
        if not toks:
            parts.append("  (no tokens in this quantile)")
            continue
        for i, t in enumerate(toks, 1):
            domain = "B" if t.get("domain_label") == 1 else "A"
            context = t.get("context", "").replace("\n", " ").strip()
            parts.append(
                f"  [{i:3d}] act={t.get('activation', 0):.3f} [{domain}] "
                f"token={repr(t.get('token', ''))}"
                f"  | context: {context[:200]}"
            )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM Judge client (SAP AI Core / Bedrock)
# ---------------------------------------------------------------------------

def _build_proxy():
    for var in ("AICORE_AUTH_URL", "AICORE_CLIENT_ID",
                "AICORE_CLIENT_SECRET", "AICORE_BASE_URL"):
        if not os.environ.get(var):
            sys.exit(f"Missing env var: {var}")
    from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
    return get_proxy_client(
        proxy_version="gen-ai-hub",
        base_url=os.environ["AICORE_BASE_URL"],
        auth_url=os.environ["AICORE_AUTH_URL"],
        client_id=os.environ["AICORE_CLIENT_ID"],
        client_secret=os.environ["AICORE_CLIENT_SECRET"],
    )


def _build_bedrock_client(proxy, model_name: str):
    from gen_ai_hub.proxy.native.amazon.clients import Session
    from botocore.config import Config
    cfg = Config(
        region_name="us-east-1",
        read_timeout=120,
        connect_timeout=30,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    return Session().client(proxy_client=proxy, model_name=model_name, config=cfg)


def _judge_one(client, feat: dict, n_tokens_total: int,
               judge_system: str, judge_user_tmpl: str, retries: int = 5) -> dict:
    user_msg = judge_user_tmpl.format(
        quantile_blocks=_format_quantile_blocks(feat.get("_quantile_tokens", {})),
    )
    last_err = "unknown"
    for attempt in range(retries):
        try:
            resp = client.converse(
                messages=[{"role": "user", "content": [{"text": user_msg}]}],
                system=[{"text": judge_system}],
                inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
            )
            text = resp["output"]["message"]["content"][0]["text"].strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            parsed = json.loads(text)
            total = int(parsed.get("judge_score",
                        parsed.get("domain_purity", 0) +
                        parsed.get("content_richness", 0) +
                        parsed.get("specificity", 0)))
            domain = parsed.get("domain_label", "").upper().strip()
            if domain not in ("A", "B", "BOTH"):
                domain = "BOTH"
            return {
                **feat,
                "domain_label":              domain,
                "judge_score":               total,
                "judge_label":               parsed.get("label", ""),
                "judge_explanation":         parsed.get("explanation", ""),
                "judge_domain_purity":       parsed.get("domain_purity", -1),
                "judge_content_richness":    parsed.get("content_richness", -1),
                "judge_specificity":         parsed.get("specificity", -1),
            }
        except json.JSONDecodeError:
            if attempt == retries - 1:
                break
            time.sleep(2 ** attempt)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(min(30 * (2 ** attempt), 300))
            elif attempt == retries - 1:
                break
            else:
                time.sleep(min(4 * (2 ** attempt), 60))
    return {
        **feat,
        "domain_label": None,
        "judge_score": -99, "judge_label": "",
        "judge_explanation": f"ERROR: {last_err}",
        "judge_domain_purity": -1, "judge_content_richness": -1, "judge_specificity": -1,
    }


# ---------------------------------------------------------------------------
# Per-feature stats
# ---------------------------------------------------------------------------

def compute_feature_stats(indices: np.ndarray, values: np.ndarray,
                          n_features: int, domain_labels: np.ndarray):
    topk     = indices.shape[1]
    n_tokens = indices.shape[0]

    act_sum   = np.zeros(n_features, dtype=np.float64)
    act_count = np.zeros(n_features, dtype=np.int64)
    act_max   = np.zeros(n_features, dtype=np.float32)

    for k in range(topk):
        np.add.at(act_sum,   indices[:, k], values[:, k].astype(np.float64))
        np.add.at(act_count, indices[:, k], 1)
        np.maximum.at(act_max, indices[:, k], values[:, k])

    mean_activation = np.zeros(n_features, dtype=np.float64)
    nz = act_count > 0
    mean_activation[nz] = act_sum[nz] / act_count[nz]

    return {
        "mean_activation": mean_activation,
        "max_activation":  act_max,
        "n_active":        act_count,
        "n_tokens":        n_tokens,
    }


# ---------------------------------------------------------------------------
# Quantile token extraction
# ---------------------------------------------------------------------------

def make_quantile_labels(n_quantiles: int) -> list:
    labels = []
    step = 1.0 / n_quantiles
    for i in range(n_quantiles):
        lo = i * step
        hi = (i + 1) * step
        labels.append((f"{int(round(lo*100))}_{int(round(hi*100))}", lo, hi))
    return labels


def extract_quantile_tokens(indices, values, feature_idxs, domain_labels,
                             all_token_ids, tokenizer,
                             quantile_labels, tokens_per_quantile, context_window):
    n_tokens = indices.shape[0]
    topk     = indices.shape[1]

    feat_set  = set(feature_idxs)
    flat_tok  = np.repeat(np.arange(n_tokens, dtype=np.int32), topk)
    flat_feat = indices.ravel().astype(np.int32)
    flat_val  = values.ravel().astype(np.float32)

    mask     = np.isin(flat_feat, list(feat_set)) & (flat_val > 0)
    sel_tok  = flat_tok[mask]
    sel_feat = flat_feat[mask]
    sel_val  = flat_val[mask]

    sort_ord = np.lexsort((sel_tok, sel_feat))
    sel_tok  = sel_tok[sort_ord]
    sel_feat = sel_feat[sort_ord]
    sel_val  = sel_val[sort_ord]

    uniq_feats, starts, counts = np.unique(sel_feat, return_index=True, return_counts=True)
    inv_index = {}
    for fi, s, c in zip(uniq_feats, starts, counts):
        inv_index[int(fi)] = (sel_tok[s:s+c], sel_val[s:s+c])

    cw  = context_window
    rng = np.random.default_rng(0)
    result = {}

    for feat_i in feature_idxs:
        tok_arr, val_arr = inv_index.get(
            feat_i, (np.array([], dtype=np.int32), np.array([], dtype=np.float32))
        )
        order = np.argsort(val_arr)[::-1]
        seen = {}
        for si in order:
            tok_i   = int(tok_arr[si])
            act_val = float(val_arr[si])
            tok_str = tokenizer.decode([int(all_token_ids[tok_i])])
            if tok_str not in seen:
                ctx_s = max(0, tok_i - cw)
                ctx_e = min(n_tokens, tok_i + cw + 1)
                seen[tok_str] = {
                    "token":        tok_str,
                    "context":      tokenizer.decode(all_token_ids[ctx_s:ctx_e].tolist()),
                    "activation":   round(act_val, 4),
                    "token_idx":    tok_i,
                    "domain_label": int(domain_labels[tok_i]),  # 1=domain B, 0=domain A
                }
        uniq = sorted(seen.values(), key=lambda x: x["activation"], reverse=True)
        n = len(uniq)

        bins = {}
        for label, lo, hi in quantile_labels:
            if n == 0:
                bins[label] = []
                continue
            start = int(n * lo)
            end   = n if hi >= 1.0 else max(start + 1, int(n * hi))
            pool  = uniq[start:end]
            k     = min(tokens_per_quantile, len(pool))
            if k <= 0:
                bins[label] = []
                continue
            chosen = rng.choice(len(pool), size=k, replace=False)
            bins[label] = [pool[j] for j in sorted(chosen)]
        result[feat_i] = bins

    return result


# ---------------------------------------------------------------------------
# Decoder norm + frequency-based classification
# ---------------------------------------------------------------------------

def compute_norm_flags(cc_ckpt: dict, n_features: int, threshold: float):
    sd = cc_ckpt["model_state_dict"]
    W_dec_A = sd["W_dec_A"]  # (n_features, d_model)
    W_dec_B = sd["W_dec_B"]

    norm_A = W_dec_A.norm(dim=1).numpy()
    norm_B = W_dec_B.norm(dim=1).numpy()

    # Paper formula (Lindsey et al. 2025):
    # Δnorm = 0.5 * (1 + (‖B‖ - ‖A‖) / max(‖A‖, ‖B‖))
    # Range [0, 1]: 0 = A-only, 0.5 = shared, 1 = B-only
    denom = np.maximum(norm_A, norm_B) + 1e-8
    delta = 0.5 * (1.0 + (norm_B - norm_A) / denom)

    flags = np.full(n_features, "BOTH", dtype=object)
    flags[delta < threshold]          = "A"   # Δnorm < 0.1 → model-A-only
    flags[delta > (1.0 - threshold)]  = "B"   # Δnorm > 0.9 → model-B-only

    freq_A = cc_ckpt.get("freq_A", None)
    freq_B = cc_ckpt.get("freq_B", None)

    return norm_A, norm_B, delta, flags, freq_A, freq_B


def compute_freq_flags(freq_A, freq_B, n_features: int, ratio_threshold: float):
    if freq_A is None or freq_B is None:
        return np.zeros(n_features), np.zeros(n_features), np.full(n_features, "BOTH", dtype=object)

    fa = freq_A.numpy() if isinstance(freq_A, torch.Tensor) else np.array(freq_A)
    fb = freq_B.numpy() if isinstance(freq_B, torch.Tensor) else np.array(freq_B)

    flags = np.full(n_features, "BOTH", dtype=object)
    eps   = 1e-8
    ratio = (fa + eps) / (fb + eps)
    flags[ratio >  ratio_threshold]        = "A"   # fires more on model A input
    flags[ratio < (1.0 / ratio_threshold)] = "B"   # fires more on model B input

    return fa, fb, flags


def compute_activation_freq(cc: CrossCoderV5, acts_a_meta_path: str,
                             acts_b_meta_path: str, n_sample: int,
                             device: torch.device):
    """Per-feature freq via half-zeroed encoding: freq_a = encode(x_a, zeros), freq_b = encode(zeros, x_b)."""

    meta_A = torch.load(acts_a_meta_path, weights_only=False, map_location="cpu")
    meta_B = torch.load(acts_b_meta_path, weights_only=False, map_location="cpu")
    n_A, d = meta_A["shape"]
    n_B, _ = meta_B["shape"]
    mmap_A = np.memmap(meta_A["mmap_path"], dtype="float32", mode="r", shape=(n_A, d))
    mmap_B = np.memmap(meta_B["mmap_path"], dtype="float32", mode="r", shape=(n_B, d))

    # Sample from the overlap of both activation arrays to avoid out-of-bounds
    n_sample = min(n_sample, n_A, n_B)
    rng = np.random.default_rng(42)
    idxs = rng.choice(min(n_A, n_B), size=n_sample, replace=False)

    n_features = cc.n_features
    freq_A = torch.zeros(n_features, device=device)
    freq_B = torch.zeros(n_features, device=device)
    batch_size = 4096

    cc.eval()
    with torch.no_grad():
        for start in range(0, n_sample, batch_size):
            batch_idxs = idxs[start:start + batch_size]
            x_A = torch.from_numpy(mmap_A[batch_idxs].copy()).float().to(device)
            x_B = torch.from_numpy(mmap_B[batch_idxs].copy()).float().to(device)
            zeros_A = torch.zeros_like(x_A)
            zeros_B = torch.zeros_like(x_B)
            f_A = cc.encode(x_A, zeros_B)   # model A input only
            f_B = cc.encode(zeros_A, x_B)   # model B input only
            freq_A += (f_A > 0).float().sum(dim=0)
            freq_B += (f_B > 0).float().sum(dim=0)

    freq_A = (freq_A / n_sample).cpu()
    freq_B = (freq_B / n_sample).cpu()
    return freq_A, freq_B


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc-acts-meta",         required=True)
    parser.add_argument("--crosscoder",            required=True)
    parser.add_argument("--dataset",               required=True)
    parser.add_argument("--model-a",               required=True,
                        help="HF ID or path of Model A — used to load tokenizer")
    parser.add_argument("--acts-a",                default=None,
                        help="Path to Model A activations .pt meta — for live freq computation")
    parser.add_argument("--acts-b",                default=None,
                        help="Path to Model B activations .pt meta — for live freq computation")
    parser.add_argument("--freq-n-sample",         type=int, default=200000)
    parser.add_argument("--domain-a-label",        default="Domain A",
                        help="Human-readable label for Domain/Model A (shown in judge prompt)")
    parser.add_argument("--domain-b-label",        default="Domain B",
                        help="Human-readable label for Domain/Model B (shown in judge prompt)")
    parser.add_argument("--out-dir",               default="./llmaaj_results")
    parser.add_argument("--judge-top-n",           type=int,   default=999999)
    parser.add_argument("--n-quantiles",           type=int,   default=10)
    parser.add_argument("--tokens-per-quantile",   type=int,   default=50)
    parser.add_argument("--context-window",        type=int,   default=10)
    parser.add_argument("--workers",               type=int,   default=8)
    parser.add_argument("--judge-model",           default="anthropic--claude-4.6-sonnet")
    parser.add_argument("--delta-norm-threshold",  type=float, default=0.1)
    parser.add_argument("--freq-ratio-threshold",  type=float, default=5.0)
    parser.add_argument("--tag",                   default=None)
    parser.add_argument("--checkpoint-every",      type=int,   default=500)
    parser.add_argument("--skip-judge",            action="store_true",
                        help="Skip LLM judge — only compute stats and save CSV")
    parser.add_argument("--existing-csv",          default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quantile_labels = make_quantile_labels(args.n_quantiles)
    quantile_cols   = [f"quantile_{lab}" for (lab, _, _) in quantile_labels]

    # Build judge prompt (parameterized by domain labels)
    judge_system   = build_judge_system(args.domain_a_label, args.domain_b_label)
    judge_user_tmpl = build_judge_user(args.domain_a_label, args.domain_b_label)

    # -----------------------------------------------------------------------
    # 1. Load CC activations meta
    # -----------------------------------------------------------------------
    print("Loading CrossCoder activations meta ...", flush=True)
    meta = torch.load(args.cc_acts_meta, map_location="cpu", weights_only=False)
    n_features = meta["n_features"]
    topk       = meta["topk"]
    print(f"  n_features={n_features:,}  topk={topk}  n_tokens={meta['n_tokens']:,}", flush=True)

    indices = np.load(meta["indices_path"])
    values  = np.load(meta["values_path"])
    # Domain labels: 1=domain B, 0=domain A
    domain_path = meta.get("domain_path") or meta.get("lang_path")  # accept legacy lang_path
    if domain_path and Path(domain_path).exists():
        domain_labels = np.load(domain_path)
        n_a = int((domain_labels == 0).sum())
        n_b = int((domain_labels == 1).sum())
        print(f"  domain labels: A={n_a:,}  B={n_b:,}", flush=True)
    else:
        print("  WARNING: no domain_path in meta — all tokens treated as Domain A", flush=True)
        domain_labels = np.zeros(meta["n_tokens"], dtype=np.uint8)

    # -----------------------------------------------------------------------
    # 2. Load CrossCoder checkpoint → norm flags + freq flags
    # -----------------------------------------------------------------------
    print("\nLoading CrossCoder checkpoint ...", flush=True)
    cc_ckpt = torch.load(args.crosscoder, map_location="cpu", weights_only=False)
    cfg     = cc_ckpt["config"]
    print(f"  arch={cfg.get('arch','?')}  d_model={cfg['d_model']}  n_features={cfg['n_features']}", flush=True)

    norm_a, norm_b, delta_norm, norm_flags, freq_A_raw, freq_B_raw = \
        compute_norm_flags(cc_ckpt, n_features, args.delta_norm_threshold)

    if (freq_A_raw is None or freq_B_raw is None) and args.acts_a and args.acts_b:
        print(f"\nComputing activation frequencies (n_sample={args.freq_n_sample:,}) ...", flush=True)
        cc_model = CrossCoderV5(cfg["d_model"], cfg["n_features"], cfg["topk"])
        cc_model.load_state_dict(cc_ckpt["model_state_dict"])
        cc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cc_model = cc_model.to(cc_device)
        freq_A_raw, freq_B_raw = compute_activation_freq(
            cc_model, args.acts_a, args.acts_b, args.freq_n_sample, cc_device
        )
        print(f"  freq_a: mean={freq_A_raw.mean():.5f}  freq_b: mean={freq_B_raw.mean():.5f}", flush=True)
        del cc_model
        torch.cuda.empty_cache()

    freq_a, freq_b, freq_flags = \
        compute_freq_flags(freq_A_raw, freq_B_raw, n_features, args.freq_ratio_threshold)

    n_a_norm  = int((norm_flags == "A").sum())
    n_b_norm  = int((norm_flags == "B").sum())
    n_both_norm = int((norm_flags == "BOTH").sum())
    print(f"  norm_flag (t={args.delta_norm_threshold}): A={n_a_norm:,}  B={n_b_norm:,}  BOTH={n_both_norm:,}", flush=True)

    n_a_freq  = int((freq_flags == "A").sum())
    n_b_freq  = int((freq_flags == "B").sum())
    n_both_freq = int((freq_flags == "BOTH").sum())
    print(f"  freq_flag (ratio>{args.freq_ratio_threshold}): A={n_a_freq:,}  B={n_b_freq:,}  BOTH={n_both_freq:,}", flush=True)

    # -----------------------------------------------------------------------
    # 3. Load tokenizer + dataset token IDs
    # -----------------------------------------------------------------------
    print("\nLoading tokenizer + dataset ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_a)

    all_token_ids_list = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") != "test":
                continue
            ids = tokenizer(rec["text"], truncation=True, max_length=2048)["input_ids"]
            all_token_ids_list.extend(ids)

    all_token_ids = np.array(all_token_ids_list, dtype=np.int32)
    print(f"  Total tokens: {len(all_token_ids):,}", flush=True)

    if len(all_token_ids) != indices.shape[0]:
        print(f"  WARNING: token count mismatch — dataset={len(all_token_ids):,}, "
              f"activations={indices.shape[0]:,}. Truncating/padding to match.", flush=True)
        target = indices.shape[0]
        if len(all_token_ids) > target:
            all_token_ids = all_token_ids[:target]
        else:
            all_token_ids = np.pad(all_token_ids, (0, target - len(all_token_ids)))

    # -----------------------------------------------------------------------
    # 4. Feature stats
    # -----------------------------------------------------------------------
    print("\nComputing feature stats ...", flush=True)
    stats = compute_feature_stats(indices, values, n_features, domain_labels)

    alive_idxs = np.where(stats["n_active"] > 0)[0].tolist()
    n_dead = n_features - len(alive_idxs)
    print(f"  Alive: {len(alive_idxs):,} / {n_features:,}  dead: {n_dead:,}", flush=True)

    judge_feature_idxs = alive_idxs[:args.judge_top_n]

    # -----------------------------------------------------------------------
    # 5. Quantile token extraction
    # -----------------------------------------------------------------------
    print(f"\nExtracting quantile tokens for {len(alive_idxs):,} features ...", flush=True)
    quantile_tokens = extract_quantile_tokens(
        indices, values, alive_idxs, domain_labels, all_token_ids,
        tokenizer, quantile_labels, args.tokens_per_quantile, args.context_window,
    )

    # -----------------------------------------------------------------------
    # 6. Build rows
    # -----------------------------------------------------------------------
    def make_row(fidx):
        qtoks = quantile_tokens.get(fidx, {})
        row = {
            "feature_idx":             fidx,
            "domain_label":            None,
            "judge_label":             None,
            "judge_explanation":       None,
            "judge_score":             None,
            "judge_domain_purity":     None,
            "judge_content_richness":  None,
            "judge_specificity":       None,
            "norm_a":                  float(norm_a[fidx]),
            "norm_b":                  float(norm_b[fidx]),
            "delta_norm":              float(delta_norm[fidx]),
            "norm_flag":               str(norm_flags[fidx]),
            "freq_a":                  float(freq_a[fidx]),
            "freq_b":                  float(freq_b[fidx]),
            "freq_flag":               str(freq_flags[fidx]),
            "n_active":                int(stats["n_active"][fidx]),
            "mean_activation":         float(stats["mean_activation"][fidx]),
            "max_activation":          float(stats["max_activation"][fidx]),
            "_quantile_tokens":        qtoks,
        }
        for (lab, _, _) in quantile_labels:
            toks = qtoks.get(lab, [])
            row[f"quantile_{lab}"] = " | ".join(repr(t["token"]) for t in toks)
        return row

    print("Building feature table ...", flush=True)
    rows = [make_row(fidx) for fidx in tqdm(alive_idxs, desc="Features")]
    row_by_fidx = {r["feature_idx"]: r for r in rows}

    # -----------------------------------------------------------------------
    # 6b. Merge judge labels from existing CSV (if provided)
    # -----------------------------------------------------------------------
    JUDGE_COLS = ["domain_label", "judge_label", "judge_explanation",
                  "judge_score", "judge_domain_purity", "judge_content_richness", "judge_specificity"]

    if args.existing_csv:
        print(f"\nMerging judge labels from existing CSV: {args.existing_csv} ...", flush=True)
        n_merged = 0
        with open(args.existing_csv, encoding="utf-8") as f:
            for old_row in csv.DictReader(f):
                fidx = int(old_row["feature_idx"])
                if fidx in row_by_fidx:
                    for col in JUDGE_COLS:
                        v = old_row.get(col, "")
                        row_by_fidx[fidx][col] = v if v != "" else None
                    n_merged += 1
        print(f"  Merged judge labels for {n_merged:,} / {len(rows):,} features.", flush=True)
        if n_merged == len(rows):
            print("  All features have judge labels — skipping LLM judge.", flush=True)
            args.skip_judge = True

    # -----------------------------------------------------------------------
    # 7. Output setup
    # -----------------------------------------------------------------------
    fieldnames = [
        "feature_idx", "domain_label", "judge_label", "judge_explanation",
        "judge_score", "judge_domain_purity", "judge_content_richness", "judge_specificity",
        "norm_a", "norm_b", "delta_norm", "norm_flag",
        "freq_a", "freq_b", "freq_flag",
        "n_active", "mean_activation", "max_activation",
    ] + quantile_cols

    hidden = {"_quantile_tokens"}
    date_str = datetime.now(LA_TZ).strftime("%Y%m%d_%H%M%S")
    tag      = f"_{args.tag}" if args.tag else ""
    csv_path = out_dir / f"crosscoder_features{tag}_{date_str}.csv"

    def _write_csv(path, rows_):
        rows_sorted = sorted(
            rows_, key=lambda r: (r.get("judge_score") if r.get("judge_score") is not None else -1),
            reverse=True,
        )
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows_sorted:
                writer.writerow({k: v for k, v in row.items() if k not in hidden})
        tmp.replace(path)  # atomic rename — avoids corrupt file on crash

    # -----------------------------------------------------------------------
    # 8. LLM Judge (optional)
    # -----------------------------------------------------------------------
    if args.skip_judge:
        print("\n--skip-judge set — writing CSV without LLM labels ...", flush=True)
        _write_csv(csv_path, rows)
        print(f"  {len(rows):,} rows → {csv_path.resolve()}", flush=True)
        print("Done.", flush=True)
        return

    print(f"\nLLM Judge: {len(judge_feature_idxs):,} features, {args.workers} workers ...", flush=True)
    proxy  = _build_proxy()
    client = _build_bedrock_client(proxy, args.judge_model)
    n_tokens_total = int(stats["n_tokens"])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_fidx = {
            executor.submit(_judge_one, client, row_by_fidx[fidx], n_tokens_total,
                            judge_system, judge_user_tmpl): fidx
            for fidx in judge_feature_idxs
        }
        done = 0
        for future in as_completed(future_to_fidx):
            fidx = future_to_fidx[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "domain_label": None, "judge_score": -99, "judge_label": "",
                    "judge_explanation": str(e), "judge_domain_purity": -1,
                    "judge_content_richness": -1, "judge_specificity": -1,
                }
            row = row_by_fidx.get(fidx)
            if row is not None:
                row["domain_label"]            = result.get("domain_label")
                row["judge_score"]             = result.get("judge_score")
                row["judge_label"]             = result.get("judge_label", "")
                row["judge_explanation"]       = result.get("judge_explanation", "")
                row["judge_domain_purity"]     = result.get("judge_domain_purity")
                row["judge_content_richness"]  = result.get("judge_content_richness")
                row["judge_specificity"]       = result.get("judge_specificity")
            done += 1
            if done % 100 == 0:
                print(f"  Judged {done}/{len(judge_feature_idxs)}", flush=True)
            if args.checkpoint_every > 0 and done % args.checkpoint_every == 0:
                _write_csv(csv_path, rows)
                print(f"  [checkpoint] {done} judged — flushed {csv_path.name}", flush=True)

    # -----------------------------------------------------------------------
    # 9. Write final CSV
    # -----------------------------------------------------------------------
    print(f"\nWriting final CSV ...", flush=True)
    _write_csv(csv_path, rows)
    print(f"  {len(rows):,} rows → {csv_path.resolve()}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
