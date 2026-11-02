"""
singlesae_salesforce_analysis.py

Single-SAE feature analysis for the Salesforce/FinTrain domain.
Identical structure to singlesae_analysis.py but with a finance-domain judge prompt.

The LLM judge scores features on how well they capture financial/Salesforce domain
knowledge (SEC filings, monetary policy, financial advisory, compliance, etc.)
vs. generic English text.

Input:
    --meta  /path/to/sae_acts_*_meta.pt   (points to _indices.npy + _values.npy)
    --sae   /path/to/sae_*.pt

Output:
    singlesae_salesforce_features_<tag>_<date>.csv

CSV columns:
    feature_idx, domain_label, judge_label, judge_explanation,
    judge_score, judge_finance_purity, judge_content_richness, judge_specificity,
    sensitivity, specificity, sens_x_spec, n_active, mean_activation, max_activation,
    quantile_0_10, quantile_10_20, ... quantile_90_100

Defaults:
    --judge-top-n          999999
    --n-quantiles          10
    --tokens-per-quantile  100
    --context-window       10
    --workers              8
    --judge-model          anthropic--claude-4.6-sonnet
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

load_dotenv()

LA_TZ = timezone(timedelta(hours=-7))


# ---------------------------------------------------------------------------
# LLM Judge — Finance / Salesforce domain
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an expert in neural network interpretability analyzing SAE (Sparse Autoencoder) features \
from a language model fine-tuned on financial text (Salesforce/FinTrain) and general English text.
You always respond with valid JSON only — no prose, no markdown, no explanation outside the JSON object.

You will be shown activating tokens for a single SAE feature, organized into activation-strength \
quantiles (0-10% = the strongest activations, 90-100% = the weakest). Each token is labeled \
[FIN] (Salesforce/finance domain text) or [EN] (general English text), with its activation \
strength and surrounding context.

The finance domain text comes from sources such as:
  - SEC filings, PCAOB and AICPA regulatory documents
  - Federal Reserve speeches and policy documents
  - Financial advisory and wealth management content
  - Hedge fund and investment compliance documents
  - Accounting standards (revenue recognition, GAAP, IFRS)
  - Stock market reports, monetary policy analysis
  - Financial instruments (ETFs, equity orders, price improvement)

Your task is to score this feature on how useful it is for understanding what the model learned \
about financial/Salesforce domain knowledge. A high score means: "this feature captures something \
meaningful and finance-specific." A low score means: "this feature is generic, domain-agnostic, \
or fires on structural noise rather than financial content."

Score the feature on these three criteria (output each as a separate integer):

  finance_purity (0-4):
    How exclusively does this feature fire on financial domain tokens?
    0 = fires predominantly on [EN] tokens, or roughly equally on both.
    1 = fires on both but noticeably more on finance.
    2 = fires mostly on finance with occasional general English tokens.
    3 = fires almost exclusively on finance, only rare [EN] exceptions.
    4 = fires ONLY on [FIN] tokens. Zero or near-zero general English activation visible.

  content_richness (0-4):
    Does this feature capture meaningful financial content rather than structural/surface tokens?
    0 = purely structural noise — BOS, EOS, padding, whitespace, newlines only.
    1 = almost entirely punctuation or single characters with no semantic content.
    2 = mostly structural with some content words mixed in; financial content is recognizable but not dominant.
    3 = mostly meaningful content — recognizable financial words, topics, or concepts dominate.
    4 = rich, specific financial semantic content — a clear finance topic cluster or domain concept \
(e.g. "SEC enforcement actions", "hedge fund custody rules", "monetary policy transmission").

  specificity (0-4):
    Can you form a clear, specific interpretation of what this feature detects?
    0 = no clear pattern, random mix of unrelated tokens with no obvious theme.
    1 = vague pattern — you can tell it's "finance-related" but nothing more specific.
    2 = clear pattern but broad (e.g. "financial regulatory terms").
    3 = moderately specific pattern with recognizable financial context.
    4 = precise, specific pattern with clear financial context \
(e.g. "SEC order-tracking rule filings" or "Federal Reserve monetary policy speeches").

  judge_score = finance_purity + content_richness + specificity  (range: 0 to 12)

Also classify the domain of this feature:
  domain_label:
    "FIN"  = fires only or almost only on finance domain tokens
    "ENG"  = fires only or almost only on general English tokens
    "BOTH" = fires meaningfully on both finance and general English tokens

Output ONLY this JSON object, nothing else:
{
  "label": "<short label max 8 words describing what this feature detects>",
  "explanation": "<2-3 sentences: what does this feature detect, in what context, why is it interesting or not>",
  "domain_label": "<FIN|ENG|BOTH>",
  "finance_purity": <0-4>,
  "content_richness": <0-4>,
  "specificity": <0-4>,
  "judge_score": <integer from 0 to 12>
}
"""

JUDGE_USER = """\
Activating tokens organized by activation-strength quantile (0-10% = strongest, 90-100% = weakest).
[FIN] = token from Salesforce/finance domain text, [EN] = token from general English text:

{quantile_blocks}

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
            lang    = "FIN" if t.get("salesforce_label") else "EN"
            context = t.get("context", "").replace("\n", " ").strip()
            parts.append(
                f"  [{i:3d}] act={t.get('activation', 0):.3f} [{lang}] "
                f"token={repr(t.get('token', ''))}"
                f"  | context: {context[:200]}"
            )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM Judge client
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


def _judge_one(client, feat: dict, n_tokens_total: int, retries: int = 5) -> dict:
    user_msg = JUDGE_USER.format(
        quantile_blocks = _format_quantile_blocks(feat.get("_quantile_tokens", {})),
    )
    last_err = "unknown"
    for attempt in range(retries):
        try:
            resp = client.converse(
                messages=[{"role": "user", "content": [{"text": user_msg}]}],
                system=[{"text": JUDGE_SYSTEM}],
                inferenceConfig={"maxTokens": 1024, "temperature": 0.2},
            )
            text = resp["output"]["message"]["content"][0]["text"].strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            parsed = json.loads(text)
            total = int(parsed.get("judge_score",
                        parsed.get("finance_purity", 0) +
                        parsed.get("content_richness", 0) +
                        parsed.get("specificity", 0)))
            lang = parsed.get("domain_label", "").upper().strip()
            if lang not in ("FIN", "ENG", "BOTH"):
                lang = "BOTH"
            return {
                **feat,
                "domain_label":              lang,
                "judge_score":               total,
                "judge_label":               parsed.get("label", ""),
                "judge_explanation":         parsed.get("explanation", ""),
                "judge_finance_purity":      parsed.get("finance_purity", -1),
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
        "judge_finance_purity": -1, "judge_content_richness": -1,
        "judge_specificity": -1,
    }


# ---------------------------------------------------------------------------
# Per-feature stats
# ---------------------------------------------------------------------------

def compute_feature_stats(indices: np.ndarray, values: np.ndarray,
                          n_features: int, token_labels: np.ndarray):
    topk = indices.shape[1]
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

    n_fin  = int((token_labels == 1).sum())
    n_pile = int((token_labels == 0).sum())

    active_on_fin  = np.zeros(n_features, dtype=np.int64)
    active_on_pile = np.zeros(n_features, dtype=np.int64)
    for k in range(topk):
        col = indices[:, k]
        np.add.at(active_on_fin,  col[token_labels == 1], 1)
        np.add.at(active_on_pile, col[token_labels == 0], 1)

    sensitivity = active_on_fin  / max(n_fin,  1)
    specificity = 1.0 - (active_on_pile / max(n_pile, 1))

    return {
        "mean_activation": mean_activation,
        "max_activation":  act_max,
        "n_active":        act_count,
        "sensitivity":     sensitivity,
        "specificity":     specificity,
        "sens_x_spec":     sensitivity * specificity,
        "n_tokens":        n_tokens,
        "n_fin":           n_fin,
        "n_pile":          n_pile,
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


def extract_quantile_tokens(indices, values, feature_idxs, token_labels,
                             all_token_ids, salesforce_token_labels, tokenizer,
                             quantile_labels, tokens_per_quantile, context_window):
    n_tokens = indices.shape[0]
    topk     = indices.shape[1]

    feat_set = set(feature_idxs)
    flat_tok  = np.repeat(np.arange(n_tokens, dtype=np.int32), topk)
    flat_feat = indices.ravel().astype(np.int32)
    flat_val  = values.ravel().astype(np.float32)

    mask = np.isin(flat_feat, list(feat_set)) & (flat_val > 0)
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
                    "token":            tok_str,
                    "context":          tokenizer.decode(all_token_ids[ctx_s:ctx_e].tolist()),
                    "activation":       round(act_val, 4),
                    "token_idx":        tok_i,
                    "salesforce_label": int(salesforce_token_labels[tok_i]),
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta",                required=True)
    parser.add_argument("--sae",                 required=True)
    parser.add_argument("--dataset",             default=None)
    parser.add_argument("--split",               default=None)
    parser.add_argument("--out-dir",             default="./singlesae_salesforce_analysis")
    parser.add_argument("--judge-top-n",         type=int, default=999999)
    parser.add_argument("--n-quantiles",         type=int, default=10)
    parser.add_argument("--tokens-per-quantile", type=int, default=100)
    parser.add_argument("--context-window",      type=int, default=10)
    parser.add_argument("--workers",             type=int, default=8)
    parser.add_argument("--judge-model",         default="anthropic--claude-4.6-sonnet")
    parser.add_argument("--tag",                 default=None)
    parser.add_argument("--checkpoint-every",    type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quantile_labels = make_quantile_labels(args.n_quantiles)
    quantile_cols   = [f"quantile_{lab}" for (lab, _, _) in quantile_labels]

    # -----------------------------------------------------------------------
    # 1. Load meta + tokenizer + dataset
    # -----------------------------------------------------------------------
    meta = torch.load(args.meta, map_location="cpu", weights_only=True)
    dataset_path = args.dataset if args.dataset else meta["dataset"]
    split        = args.split   if args.split   else meta["split"]
    n_features   = meta["n_features"]

    print(f"Meta:    {args.meta}")
    print(f"SAE:     {args.sae}")
    print(f"Model:   {meta['model']}")
    print(f"Dataset: {dataset_path}  (split: {split})")
    print(f"Features: {n_features:,}  Quantiles: {args.n_quantiles} "
          f"({args.tokens_per_quantile} tok/bin)")

    print("\n=== Loading tokenizer + dataset ===")
    tokenizer = AutoTokenizer.from_pretrained(meta["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_token_ids_list = []
    token_labels_list  = []          # 1 = finance/salesforce, 0 = english
    salesforce_labels_list = []

    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if split != "all" and rec.get("split") != split:
                continue
            ids   = tokenizer(rec["text"], truncation=True, max_length=2048)["input_ids"]
            label = 1 if rec.get("salesforce_label") else 0
            all_token_ids_list.extend(ids)
            token_labels_list.extend([label] * len(ids))
            salesforce_labels_list.extend([label] * len(ids))

    token_labels          = np.array(token_labels_list,     dtype=np.int8)
    salesforce_token_labels = np.array(salesforce_labels_list, dtype=np.int8)
    all_token_ids         = np.array(all_token_ids_list,    dtype=np.int32)
    print(f"  Tokens: {len(token_labels):,}  FIN: {int(token_labels.sum()):,}  "
          f"EN: {int((token_labels==0).sum()):,}")

    # -----------------------------------------------------------------------
    # 2. Load activations + stats
    # -----------------------------------------------------------------------
    print("\n=== Loading activations + computing stats ===")
    indices = np.load(meta["indices_path"])
    values  = np.load(meta["values_path"])
    print(f"  indices: {indices.shape}  values: {values.shape}")

    assert indices.shape[0] == len(token_labels), (
        f"Token count mismatch: activations have {indices.shape[0]} tokens, "
        f"dataset yields {len(token_labels)} for split={split}"
    )

    stats = compute_feature_stats(indices, values, n_features, token_labels)
    n_tokens_total = int(stats["n_tokens"])

    alive_idxs = np.where(stats["n_active"] > 0)[0].tolist()
    n_dead = n_features - len(alive_idxs)
    print(f"  Alive: {len(alive_idxs):,} / {n_features:,}  (dead: {n_dead:,})")

    judge_feature_idxs = alive_idxs[:args.judge_top_n]

    # -----------------------------------------------------------------------
    # 3. Quantile token extraction
    # -----------------------------------------------------------------------
    print(f"\n=== Extracting quantile tokens for {len(alive_idxs):,} features ===")
    quantile_tokens = extract_quantile_tokens(
        indices, values, alive_idxs, token_labels, all_token_ids,
        salesforce_token_labels, tokenizer, quantile_labels,
        args.tokens_per_quantile, args.context_window,
    )

    # -----------------------------------------------------------------------
    # 4. Build rows
    # -----------------------------------------------------------------------
    def make_row(fidx):
        qtoks = quantile_tokens.get(fidx, {})
        row = {
            "feature_idx":           fidx,
            "domain_label":          None,
            "judge_label":           None,
            "judge_explanation":     None,
            "judge_score":           None,
            "judge_finance_purity":  None,
            "judge_content_richness": None,
            "judge_specificity":     None,
            "sensitivity":     float(stats["sensitivity"][fidx]),
            "specificity":     float(stats["specificity"][fidx]),
            "sens_x_spec":     float(stats["sens_x_spec"][fidx]),
            "n_active":        int(stats["n_active"][fidx]),
            "mean_activation": float(stats["mean_activation"][fidx]),
            "max_activation":  float(stats["max_activation"][fidx]),
            "_quantile_tokens": qtoks,
        }
        for (lab, _, _) in quantile_labels:
            toks = qtoks.get(lab, [])
            row[f"quantile_{lab}"] = ", ".join(repr(t["token"]) for t in toks)
        return row

    print("\n=== Building table ===")
    rows = [make_row(fidx) for fidx in tqdm(alive_idxs, desc="Features")]
    row_by_fidx = {r["feature_idx"]: r for r in rows}

    # -----------------------------------------------------------------------
    # 5. Output paths + writer
    # -----------------------------------------------------------------------
    fieldnames = [
        "feature_idx", "domain_label", "judge_label", "judge_explanation",
        "judge_score", "judge_finance_purity", "judge_content_richness", "judge_specificity",
        "sensitivity", "specificity", "sens_x_spec",
        "n_active", "mean_activation", "max_activation",
    ] + quantile_cols

    hidden = {"_quantile_tokens"}
    date_str = datetime.now(LA_TZ).strftime("%Y%m%d_%H%M%S")
    tag      = f"_{args.tag}" if args.tag else ""
    csv_path = out_dir / f"singlesae_salesforce_features{tag}_{date_str}.csv"

    def _write_csv(path, rows_):
        rows_sorted = sorted(
            rows_, key=lambda r: (r.get("judge_score") if r.get("judge_score") is not None else -1),
            reverse=True,
        )
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows_sorted:
                writer.writerow({k: v for k, v in row.items() if k not in hidden})

    # -----------------------------------------------------------------------
    # 6. LLM Judge
    # -----------------------------------------------------------------------
    print(f"\n=== LLM Judge on {len(judge_feature_idxs):,} features "
          f"({args.workers} workers) ===")
    proxy  = _build_proxy()
    client = _build_bedrock_client(proxy, args.judge_model)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_fidx = {
            executor.submit(_judge_one, client, row_by_fidx[fidx], n_tokens_total): fidx
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
                    "judge_explanation": str(e), "judge_finance_purity": -1,
                    "judge_content_richness": -1, "judge_specificity": -1,
                }
            row = row_by_fidx.get(fidx)
            if row is not None:
                row["domain_label"]           = result.get("domain_label")
                row["judge_score"]            = result.get("judge_score")
                row["judge_label"]            = result.get("judge_label", "")
                row["judge_explanation"]      = result.get("judge_explanation", "")
                row["judge_finance_purity"]   = result.get("judge_finance_purity")
                row["judge_content_richness"] = result.get("judge_content_richness")
                row["judge_specificity"]      = result.get("judge_specificity")
            done += 1
            if done % 100 == 0:
                print(f"  Judged {done}/{len(judge_feature_idxs)}", flush=True)
            if args.checkpoint_every > 0 and done % args.checkpoint_every == 0:
                _write_csv(csv_path, rows)
                print(f"  [checkpoint] {done} judged — flushed {csv_path.name}", flush=True)

    # -----------------------------------------------------------------------
    # 7. Write final CSV
    # -----------------------------------------------------------------------
    print(f"\n=== Writing CSV ===")
    _write_csv(csv_path, rows)
    print(f"  {len(rows):,} rows → {csv_path.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
