"""
paper_vllm_confidence_interval_fintrain.py

Identical to paper_vllm_confidence_interval.py but uses Salesforce/FinTrain domain prompts
instead of SAP prompts. Intended for the Llama-Fin-8b / Meta-Llama-3-8B-Instruct experiment.
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


class ContentFilteredError(Exception):
    """Raised when SAP AI Core's content filter blocks a prompt (stopReason=content_filtered).
    The caller should re-sample the quantile tokens to build a different prompt."""
    pass


class BedrockExplainer:
    """Claude via SAP AI Core (Bedrock) — drop-in SOTA explainer."""

    def __init__(self, model_name: str = "anthropic--claude-4.7-opus"):
        from dotenv import load_dotenv
        load_dotenv()
        import os
        for var in ("AICORE_AUTH_URL", "AICORE_CLIENT_ID", "AICORE_CLIENT_SECRET", "AICORE_BASE_URL"):
            if not os.environ.get(var):
                raise EnvironmentError(f"Missing env var: {var}. Source .env before running with --sota.")
        self.model_name = model_name
        self._build_client()

    def _build_client(self):
        import os
        from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
        from gen_ai_hub.proxy.native.amazon.clients import Session
        from botocore.config import Config
        print("[Bedrock-diag] calling get_proxy_client()", flush=True)
        proxy = get_proxy_client(
            proxy_version="gen-ai-hub",
            base_url=os.environ["AICORE_BASE_URL"],
            auth_url=os.environ["AICORE_AUTH_URL"],
            client_id=os.environ["AICORE_CLIENT_ID"],
            client_secret=os.environ["AICORE_CLIENT_SECRET"],
        )
        print("[Bedrock-diag] get_proxy_client() returned, calling Session().client()", flush=True)
        cfg = Config(region_name="us-east-1", read_timeout=120, connect_timeout=30,
                     retries={"max_attempts": 1, "mode": "standard"})
        self.client = Session().client(proxy_client=proxy, model_name=self.model_name, config=cfg)
        print("[Bedrock-diag] Session().client() returned", flush=True)

    def generate_text(self, system: str, user: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        import concurrent.futures
        attempt = 0
        empty_attempts = 0
        MAX_EMPTY = 15
        while True:
            attempt += 1
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(self.client.converse,
                        messages=[{"role": "user", "content": [{"text": user}]}],
                        system=[{"text": system}],
                        inferenceConfig={"maxTokens": max_tokens},
                    )
                    resp = future.result(timeout=150)
                content = resp.get("output", {}).get("message", {}).get("content", [])
                if content and content[0].get("text"):
                    return content[0]["text"].strip()
                # Empty response: log stopReason to distinguish content-filter refusal
                # (permanent for this prompt) from transient API hiccup.
                empty_attempts += 1
                stop_reason = resp.get("stopReason", "?")
                # content_filtered is prompt-specific: raise so the caller can re-sample
                # the quantile tokens and build a genuinely different prompt.
                if stop_reason == "content_filtered":
                    raise ContentFilteredError()
                if empty_attempts >= MAX_EMPTY:
                    print(f"  [BedrockExplainer] empty response {empty_attempts}x (stopReason={stop_reason}), GIVING UP -> returning empty description", flush=True)
                    return ""
                wait = min(300, 30 * attempt)
                print(f"  [BedrockExplainer] empty response (attempt {attempt}, empty {empty_attempts}/{MAX_EMPTY}, stopReason={stop_reason}), rebuilding client and retrying in {wait}s...", flush=True)
                self._build_client()
            except ContentFilteredError:
                raise
            except concurrent.futures.TimeoutError:
                wait = min(300, 30 * attempt)
                print(f"  [BedrockExplainer] timeout (attempt {attempt}), rebuilding client and retrying in {wait}s...", flush=True)
                self._build_client()
            except Exception as e:
                wait = min(300, 30 * attempt)
                print(f"  [BedrockExplainer] error (attempt {attempt}): {e}, rebuilding client and retrying in {wait}s...", flush=True)
                self._build_client()
            time.sleep(wait)


from paper_experiment import (
    load_sae, load_activations, extract_all_tokens_for_feature, _load_all_token_ids,
    load_llm, register_hook,
    sample_from_quantiles, sample_control_sentences, parse_sentence_list,
    compute_generative_accuracy, compute_cosy_scores,
    load_sap_label_mask,
)
from vllm_router import VLLMRouter
from qdrant_rag import QdrantRAG

from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Fintrain / Salesforce domain prompts
# ---------------------------------------------------------------------------
_EXPLAINER_SYSTEM_BASE = (
    "You are an expert in neural network interpretability analyzing SAE (Sparse Autoencoder) "
    "features from a language model. "
    "You will be shown activating tokens for a single SAE feature, "
    "organized by activation strength quantile (strongest to weakest). "
    "Your task is to write a concise natural-language description of what this feature detects: "
    "what semantic concept or domain-specific content causes it to activate. "
    "IMPORTANT: Focus exclusively on semantic and conceptual patterns — the underlying topic, domain, "
    "or meaning that unifies the activating tokens. "
    "Do NOT describe syntactic or surface-level patterns such as punctuation marks (periods, commas, parentheses), "
    "isolated digits, or whitespace, even if these appear as high-activation tokens. "
    "Such tokens are often incidental and do not reflect the true semantic signal of the feature. "
    "If the top-quantile tokens appear syntactic, look at the surrounding context and the full distribution "
    "of tokens across all quantiles to identify the underlying semantic concept. "
    "Pay attention to the quantile labels — tokens in '0-5%' are the strongest activators, "
    "tokens in '60-100%' activate weakly. Be specific and grounded in the examples shown. "
    "Output ONLY the description — no preamble, no JSON, no bullet points."
)

EXPLAINER_SYSTEM_RAG_AWARE = _EXPLAINER_SYSTEM_BASE + (
    " If a <retrieved_context> block is provided in the user message, you may use it as "
    "additional background information about the domain. This context contains passages from "
    "the training corpus that are related to the feature's activating tokens — it can help you "
    "enrich your description with domain-specific terminology and concepts relevant to the feature. "
    "However, do not let the retrieved context override the evidence from the activating tokens themselves."
)

EXPLAINER_SYSTEM = _EXPLAINER_SYSTEM_BASE

EXPLAINER_USER_TMPL = """\
Activating tokens organized by quantile (0-5% = strongest, 60-100% = weakest).
Format: [quantile] token=<token> | context: <surrounding text>

{token_block}

Write a concise description of what this SAE feature detects."""

GENERATION_SYSTEM = (
    "You are an expert in neural network interpretability. "
    "You will be given a description of an SAE (Sparse Autoencoder) feature — the semantic concept "
    "or pattern it detects. Based ONLY on this description, generate sentences that would strongly "
    "activate this feature. Each sentence should naturally contain the concept described. "
    "Output ONLY a JSON array of strings, one sentence per element, "
    "e.g.: [\"sentence one.\", \"sentence two.\", ...]. "
    "No explanation, no prose, just the JSON array."
)

GENERATION_USER_TMPL = """\
Feature description:
{description}

Generate a JSON array of {n_tokens} sentences that you predict will strongly activate this feature."""

JUDGE_SYSTEM_PRED = (
    "You are an expert in neural network interpretability analyzing SAE (Sparse Autoencoder) "
    "features. You will be given a description of what an SAE feature detects, and a single token "
    "with its surrounding context. Your task is to predict how strongly this feature activates on "
    "that token, on a scale from 0 to 9 (0 = no activation, 9 = maximum activation). "
    "Output ONLY a single digit between 0 and 9. No explanation, no punctuation."
)

JUDGE_USER_PRED_TMPL = """\
Feature description:
{description}

Token: {token}
Context: {context}

How strongly does this feature activate on this token? Answer with a single digit 0-9."""

JUDGE_SYSTEM_DETECTION = (
    "You are an expert in neural network interpretability analyzing SAE (Sparse Autoencoder) "
    "features. You will be given a description of what an SAE feature detects, and a text context. "
    "Your task is to decide whether this feature is active in the given context. "
    "Output ONLY 'Y' (yes, the feature is active) or 'N' (no, the feature is not active). "
    "No explanation, no punctuation, just Y or N."
)

JUDGE_USER_DETECTION_TMPL = """\
Feature description:
{description}

Context: {context}

Is this feature active in the context above? Answer Y or N."""

JUDGE_SYSTEM_FUZZING = (
    "You are an intelligent and meticulous linguistics researcher. "
    "You will be given a certain feature of text, such as 'male pronouns' or 'text with negative sentiment'. "
    "You will be given a few examples of text that contain this feature. "
    "Portions of the sentence which strongly represent this feature are between tokens << and >>. "
    "Some examples might be mislabeled. Your task is to determine if every single token within << and >> is correctly "
    "labeled. Consider that all provided examples could be correct, none of the examples could be correct, or a mix. "
    "An example is only correct if every marked token is representative of the feature. "
    "For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled. "
    "You must return your response in a valid Python list. Do not return anything else besides a Python list."
)

JUDGE_USER_FUZZING_TMPL = """\
feature interpretation: {description}

Text examples:

{examples}"""


# ---------------------------------------------------------------------------
# Helpers: build prompts
# ---------------------------------------------------------------------------
def _build_explainer_user(feature_idx, quantile_tokens, rag_context=None):
    lines = []
    for t in quantile_tokens:
        ctx = t.get("context", "").replace("\n", " ").strip()
        lines.append(f"  [{t['quantile']:8s}] token={repr(t['token'])} | context: {ctx[:200]}")
    user = EXPLAINER_USER_TMPL.format(feature_idx=feature_idx,
                                       token_block="\n".join(lines))
    if rag_context:
        user = f"<retrieved_context>\n{rag_context}\n</retrieved_context>\n\n" + user
    return user


def _chat_prompt(system, user, rag_context=None):
    if rag_context:
        user = f"<retrieved_context>\n{rag_context}\n</retrieved_context>\n\n" + user
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


# ---------------------------------------------------------------------------
# Helpers: sample held-out tokens with variable seed
# ---------------------------------------------------------------------------
def _sample_held_out(all_tokens, all_token_ids, tokenizer, indices, values,
                     feature_idx, n_per_quantile, context_window, seed):
    rng = np.random.default_rng(seed)
    held_out = sample_from_quantiles(all_tokens, n_per_quantile, rng=rng)
    held_out_idxs = {t["token_idx"] for t in held_out if "token_idx" in t}

    n_tokens = indices.shape[0]
    flat_feat = indices.ravel()
    active_positions = set(
        np.repeat(np.arange(n_tokens, dtype=np.int32),
                  indices.shape[1])[flat_feat == feature_idx].tolist()
    )
    zero_rng = np.random.default_rng(seed + 1)
    candidates = zero_rng.choice(n_tokens, size=min(10000, n_tokens), replace=False)
    seen_strs = {t["token"] for t in held_out}
    zero_tokens = []
    for pos in candidates:
        if len(zero_tokens) >= n_per_quantile:
            break
        if int(pos) in active_positions:
            continue
        tok_str = tokenizer.decode([int(all_token_ids[int(pos)])])
        if tok_str in seen_strs or not tok_str.strip():
            continue
        ctx_s = max(0, int(pos) - context_window)
        ctx_e = min(n_tokens, int(pos) + context_window + 1)
        context = tokenizer.decode(all_token_ids[ctx_s:ctx_e].tolist())
        zero_tokens.append({"token": tok_str, "context": context,
                             "activation": 0.0, "normalized_score": 0.0,
                             "quantile": "zero"})
        seen_strs.add(tok_str)
    held_out.extend(zero_tokens)
    return held_out, held_out_idxs


# ---------------------------------------------------------------------------
# Endpoint judge helpers
# ---------------------------------------------------------------------------
def _endpoint_pred_acc(router, description, held_out, rag_context):
    msgs = [_chat_prompt(JUDGE_SYSTEM_PRED,
                         JUDGE_USER_PRED_TMPL.format(
                             description=description, token=repr(t["token"]),
                             context=t["context"][:300].replace("\n", " ")),
                         rag_context)
            for t in held_out]
    resps = router.chat_many(msgs, max_tokens=1, temperature=0.0,
                             logprobs=True, top_logprobs=20,
                             chat_template_kwargs={"enable_thinking": False})
    preds, gts = [], []
    for t, r in zip(held_out, resps):
        preds.append(_expected_digit(r))
        gts.append(t["normalized_score"])
    if len(preds) < 2 or len(set(gts)) < 2:
        return 0.0
    corr, _ = pearsonr(gts, preds)
    return float(corr)


def _expected_digit(resp):
    try:
        lp = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (TypeError, KeyError, IndexError):
        return 0.0
    digit_logprob = {}
    for item in lp:
        tok = item["token"].strip()
        if len(tok) == 1 and tok in "0123456789":
            digit_logprob[int(tok)] = item["logprob"]
    if not digit_logprob:
        return 0.0
    logits = torch.tensor([digit_logprob.get(i, -1e9) for i in range(10)])
    probs = torch.softmax(logits, dim=0)
    return sum(i * probs[i].item() for i in range(10))


def _endpoint_detection(router, description, held_out, rag_context):
    msgs, labels = [], []
    for t in held_out:
        q = t["quantile"]
        label = 1 if q in ("0-5%", "5-15%") else (0 if q == "zero" else None)
        labels.append(label)
        msgs.append(_chat_prompt(
            JUDGE_SYSTEM_DETECTION,
            JUDGE_USER_DETECTION_TMPL.format(
                description=description,
                context=t["context"][:400].replace("\n", " ")),
            rag_context))
    resps = router.chat_many(msgs, max_tokens=1, temperature=0.0,
                             logprobs=True, top_logprobs=20,
                             chat_template_kwargs={"enable_thinking": False})
    scores = [_p_yes(r) for r in resps]
    binary = [(s, l) for s, l in zip(scores, labels) if l is not None]
    if len(binary) < 2 or len({l for _, l in binary}) < 2:
        return 0.5
    try:
        return float(roc_auc_score([b[1] for b in binary], [b[0] for b in binary]))
    except Exception:
        return 0.5


def _p_yes(resp):
    try:
        lp = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (TypeError, KeyError, IndexError):
        return 0.5
    yl = nl = None
    for item in lp:
        tok = item["token"].strip()
        if tok == "Y" and yl is None:
            yl = item["logprob"]
        elif tok == "N" and nl is None:
            nl = item["logprob"]
    if yl is None and nl is None:
        return 0.5
    yl = yl if yl is not None else -1e9
    nl = nl if nl is not None else -1e9
    m = max(yl, nl)
    ey, en = math.exp(yl - m), math.exp(nl - m)
    return ey / (ey + en)


def _endpoint_fuzzing(router, description, held_out, rag_context, batch_size=5):
    binary = []
    for t in held_out:
        q = t["quantile"]
        label = 1 if q in ("0-5%", "5-15%") else (0 if q == "zero" else None)
        if label is None:
            continue
        ctx = t["context"][:400].replace("\n", " ")
        tok = t["token"]
        marked = ctx.replace(tok, f"<<{tok}>>", 1) if tok in ctx else f"<<{tok}>> {ctx}"
        binary.append({"label": label, "marked": marked})
    if not binary:
        return 0.5
    rng = np.random.default_rng(0)
    shuffled = [binary[i] for i in rng.permutation(len(binary)).tolist()]
    msgs, batches = [], []
    for start in range(0, len(shuffled), batch_size):
        batch = shuffled[start:start + batch_size]
        examples = "\n\n".join(f"Example {i}: {b['marked']}" for i, b in enumerate(batch))
        msgs.append(_chat_prompt(JUDGE_SYSTEM_FUZZING,
                                 JUDGE_USER_FUZZING_TMPL.format(
                                     description=description, examples=examples),
                                 rag_context))
        batches.append(batch)
    resps = router.chat_many(msgs, max_tokens=64, temperature=0.0,
                             chat_template_kwargs={"enable_thinking": False})
    correct = total = 0
    for batch, r in zip(batches, resps):
        preds = _parse_fuzzing(r, len(batch))
        for b, p in zip(batch, preds):
            total += 1
            if (1 if p >= 0.5 else 0) == b["label"]:
                correct += 1
    return correct / total if total else 0.5


def _parse_fuzzing(resp, n):
    if resp is None or "error" in (resp or {}):
        return [0.5] * n
    txt = resp["choices"][0]["message"]["content"].strip()
    m = re.search(r"\[[\s\d,]+\]", txt)
    if m:
        try:
            vals = json.loads(m.group(0))
            if isinstance(vals, list) and len(vals) == n:
                return [float(v) for v in vals]
        except (json.JSONDecodeError, ValueError):
            pass
    digits = re.findall(r"\b([01])\b", txt)
    if len(digits) == n:
        return [float(d) for d in digits]
    return [0.5] * n


def _gen_sentences_endpoint(router, description, n_tokens, max_tokens, temperature,
                             rag_context=None):
    msgs = [_chat_prompt(GENERATION_SYSTEM,
                         GENERATION_USER_TMPL.format(description=description,
                                                      n_tokens=n_tokens),
                         rag_context)]
    resps = router.chat_many(msgs, max_tokens=max_tokens, temperature=temperature,
                             chat_template_kwargs={"enable_thinking": False})
    r = resps[0]
    if r is None or "error" in (r or {}):
        return []
    return parse_sentence_list(r["choices"][0]["message"]["content"].strip())


# ---------------------------------------------------------------------------
# One metric run
# ---------------------------------------------------------------------------
def _t(label, t0):
    print(f"    [time] {label}: {time.time()-t0:.2f}s")


def _run_metrics(suffix, router, description, sentences, held_out, threshold,
                 feature_idx, gen_model, gen_tok, sae, sae_norm, meta,
                 hook_store, all_token_ids, device, args, rag_context,
                 ctrl_rng_seed):
    result = {}

    t = time.time()
    if sentences:
        gen_acc, _, syn_mean = compute_generative_accuracy(
            sentences, feature_idx, gen_model, gen_tok, sae, sae_norm,
            meta["layer"], hook_store, threshold, device)
        _t(f"{suffix}/gen_acc", t); t = time.time()
        ctrl = sample_control_sentences(
            all_token_ids, gen_tok,
            n_sentences=max(len(sentences), 20),
            context_len=50, rng=np.random.default_rng(ctrl_rng_seed))
        cosy_auc, cosy_mad, _, _ = compute_cosy_scores(
            sentences, ctrl, feature_idx, gen_model, gen_tok, sae, sae_norm,
            meta["layer"], hook_store, device, cached_syn_mean_acts=syn_mean)
        _t(f"{suffix}/cosy", t)
    else:
        gen_acc = cosy_auc = cosy_mad = float("nan")
        print(f"    [time] {suffix}/gen_acc+cosy: skipped (no sentences)")

    result[f"gen_acc_{suffix}"]         = None if math.isnan(gen_acc)  else round(gen_acc,  4)
    result[f"cosy_auc_{suffix}"]        = None if math.isnan(cosy_auc) else round(cosy_auc, 4)
    result[f"cosy_mad_{suffix}"]        = None if math.isnan(cosy_mad) else round(cosy_mad, 4)

    t = time.time()
    result[f"pred_acc_{suffix}"]        = round(_endpoint_pred_acc(
        router, description, held_out, rag_context), 4)
    _t(f"{suffix}/pred_acc", t)

    t = time.time()
    result[f"detection_auroc_{suffix}"] = round(_endpoint_detection(
        router, description, held_out, rag_context), 4)
    _t(f"{suffix}/detection", t)

    t = time.time()
    result[f"fuzzing_acc_{suffix}"]     = round(_endpoint_fuzzing(
        router, description, held_out, rag_context), 4)
    _t(f"{suffix}/fuzzing", t)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", type=int, required=True)
    p.add_argument("--meta",     required=True)
    p.add_argument("--sae-ckpt", required=True)
    p.add_argument("--generation-model", required=True)

    p.add_argument("--explainer-endpoint",    default=None)
    p.add_argument("--explainer-served-model", default=None)
    p.add_argument("--dummy-descriptions", action="store_true", default=False,
                   help="Skip explainer entirely; use fixed dummy descriptions ['', '\\n', '\\t'] "
                        "as a no-description baseline.")
    p.add_argument("--sota", action="store_true", default=False)

    p.add_argument("--cpt-endpoints",   nargs="+", required=True)
    p.add_argument("--cpt-served-model", required=True)

    p.add_argument("--qwen-endpoints",   nargs="+", required=True)
    p.add_argument("--qwen-served-model", required=True)

    p.add_argument("--judge-mode", default="base_rag",
                   choices=["qwen_rag", "base_rag", "base_norag"])

    p.add_argument("--qdrant-url",        default="http://cpu-dy-c59xlarge-3:6335")
    p.add_argument("--qdrant-collection", default="fintrain_chunks")
    p.add_argument("--rag-top-k",         type=int, default=5)
    p.add_argument("--rag-aware-prompt",   action="store_true",
                   help="Append RAG-context usage instruction to the explainer system prompt")
    p.add_argument("--explainer-rag",     action="store_true")
    p.add_argument("--sap-only-rag",      action="store_true",
                   help="When building the RAG query, use only domain-labeled token contexts "
                        "(salesforce_label==1), filtering out general-English tokens from the mixed corpus")

    p.add_argument("--n-quantile-samples", type=int, default=1)
    p.add_argument("--n-desc-samples",     type=int, default=1)
    p.add_argument("--n-metric-samples",   type=int, default=1)

    p.add_argument("--explainer-tokens-per-quantile", type=int, default=50)
    p.add_argument("--held-out-per-quantile",         type=int, default=5)
    p.add_argument("--context-window",                type=int, default=10)
    p.add_argument("--gen-tokens",                    type=int, default=20)
    p.add_argument("--explainer-max-tokens",          type=int, default=256)
    p.add_argument("--generation-max-tokens",         type=int, default=256)
    p.add_argument("--temperature",                   type=float, default=0.7)
    p.add_argument("--min-active",  type=int, default=0,
                   help="Skip features with fewer than N activating tokens after domain filtering. "
                        "Default 0 (disabled) — set to e.g. 100 to restore the old behaviour.")
    p.add_argument("--per-instance", type=int, default=32)

    p.add_argument("--output",  default=None)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not args.sota and not args.dummy_descriptions and (args.explainer_endpoint is None or args.explainer_served_model is None):
        p.error("--explainer-endpoint and --explainer-served-model are required unless --sota or --dummy-descriptions is set.")

    assert args.temperature > 0

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent.parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / f"bootstrap_fintrain_{ts}.json")

    device = torch.device(args.device)
    T_TOTAL = args.n_quantile_samples * args.n_desc_samples * args.n_metric_samples
    n_features = len(args.features)
    print(f"\n=== paper_vllm_confidence_interval_fintrain.py ===")
    print(f"Features:            {n_features}")
    print(f"Quantile samples:    {args.n_quantile_samples}")
    print(f"Description samples: {args.n_desc_samples}")
    print(f"Metric samples:      {args.n_metric_samples}")
    print(f"Records total:       {n_features} × {T_TOTAL} = {n_features * T_TOTAL}")
    print(f"Temperature:         {args.temperature}")
    print(f"Judge mode:          {args.judge_mode}")
    print(f"Output:              {args.output}")

    t0 = time.time()
    print("\n[setup] Loading SAE activation data ...")
    indices, values, meta = load_activations(args.meta)

    print("[setup] Loading SAE ...")
    sae, sae_norm = load_sae(args.sae_ckpt, device)

    print("[setup] Loading local generation model (CPT, for SAE hook) ...")
    gen_model, gen_tok = load_llm(args.generation_model, device)
    all_token_ids = _load_all_token_ids(meta["dataset"], gen_tok, meta["n_tokens"])
    sap_label_mask = None
    if args.sap_only_rag:
        print("[setup] Loading domain-label mask for RAG query filtering ...")
        sap_label_mask = load_sap_label_mask(meta["dataset"], gen_tok, meta["n_tokens"])
        n_dom = int(sap_label_mask.sum())
        print(f"[setup] Domain tokens: {n_dom}/{meta['n_tokens']} ({100*n_dom/meta['n_tokens']:.1f}%)")
    hook_handle, hook_store = register_hook(gen_model, meta["layer"])

    use_rag = args.judge_mode in ("qwen_rag", "base_rag")
    use_explainer_rag = args.explainer_rag
    EXPLAINER_SYSTEM = EXPLAINER_SYSTEM_RAG_AWARE if args.rag_aware_prompt else _EXPLAINER_SYSTEM_BASE
    rag_client = None
    if use_rag or use_explainer_rag:
        print(f"[setup] Connecting to Qdrant RAG ({args.qdrant_url}) ...")
        rag_client = QdrantRAG(
            url=args.qdrant_url,
            collection=args.qdrant_collection,
        )
        print(f"[setup] Qdrant RAG ready (collection={args.qdrant_collection})", flush=True)

    setup_time = time.time() - t0
    print(f"[setup] Done in {setup_time:.1f}s. Hook on layer {meta['layer']}.\n", flush=True)

    bedrock_explainer = None
    if args.dummy_descriptions:
        print("[setup] Dummy-description mode — no explainer loaded.", flush=True)
        explainer_router = None
    elif args.sota:
        print("[setup] Using BedrockExplainer (anthropic--claude-4.7-opus via SAP AI Core) ...", flush=True)
        import faulthandler, sys
        faulthandler.dump_traceback_later(30, repeat=True, file=sys.stderr)
        bedrock_explainer = BedrockExplainer("anthropic--claude-4.7-opus")
        faulthandler.cancel_dump_traceback_later()
        print("[setup-diag] BedrockExplainer() returned", flush=True)
        explainer_router = None
    else:
        explainer_router = VLLMRouter(
            [args.explainer_endpoint], args.explainer_served_model,
            per_instance=args.per_instance, policy="fill")
    cpt_router = VLLMRouter(
        args.cpt_endpoints, args.cpt_served_model,
        per_instance=args.per_instance, policy="fill")
    qwen_router = VLLMRouter(
        args.qwen_endpoints, args.qwen_served_model,
        per_instance=args.per_instance, policy="fill")

    all_records = []
    valid_features = [f for f in args.features if 0 <= f < meta["n_features"]]
    t_loop = time.time()

    # ------------------------------------------------------------------
    # SOTA two-phase approach: generate ALL descriptions first (Bedrock),
    # then judge with vLLM uninterrupted. This avoids long idle gaps on
    # vLLM connections caused by slow Bedrock calls, which cause the proxy
    # to silently hang on the next vLLM request.
    # For non-SOTA (vLLM explainer), descriptions are generated inline as before.
    # ------------------------------------------------------------------
    desc_cache = {}   # desc_cache[fi][q_s] = {"descriptions": [...], "held_out": ...,
                      #                         "threshold": ..., "all_tokens": ...,
                      #                         "expl_rag_ctx_pre": ...}

    if args.dummy_descriptions:
        DUMMY_DESCS = ["", "\n", "\t"]
        print(f"\n{'='*60}")
        print(f"PHASE 1 (dummy): Filling dummy descriptions for {len(valid_features)} features")
        print(f"{'='*60}")
        t_phase1 = time.time()
        for feat_i, fi in enumerate(valid_features):
            all_tokens = extract_all_tokens_for_feature(
                indices=indices, values=values, feature_idx=fi,
                dataset_path=meta["dataset"], tokenizer=gen_tok,
                context_window=args.context_window, all_token_ids=all_token_ids,
                token_mean_act=None, alpha_filter=False,
                domain_mask=sap_label_mask,
            )
            if len(all_tokens) < args.min_active:
                print(f"  [Phase1-dummy] Feature {fi} SKIPPED: n_active={len(all_tokens)} < {args.min_active}")
                continue
            top10_max = max((t["activation"] for t in all_tokens[:10]), default=0.0)
            threshold = 0.5 * top10_max
            desc_cache[fi] = {"threshold": threshold, "all_tokens": all_tokens, "quantiles": {}}
            for q_s in range(args.n_quantile_samples):
                h_seed = fi * 1000 + q_s + 500
                held_out, held_out_idxs = _sample_held_out(
                    all_tokens, all_token_ids, gen_tok, indices, values,
                    fi, args.held_out_per_quantile, args.context_window, h_seed)
                desc_cache[fi]["quantiles"][q_s] = {
                    "descriptions": list(DUMMY_DESCS),
                    "held_out": held_out,
                    "expl_rag_ctx_pre": None,
                    "explainer_pool": None,
                }
        print(f"\n[Phase1-dummy] Done in {(time.time()-t_phase1)/60:.1f}min. "
              f"Dummy descriptions set for {len(desc_cache)} features.\n")

    elif bedrock_explainer is not None:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Generating all descriptions via Bedrock ({len(valid_features)} features)")
        print(f"{'='*60}")
        t_phase1 = time.time()
        for feat_i, fi in enumerate(valid_features):
            print(f"\n[Phase1] Feature {fi}  ({feat_i+1}/{len(valid_features)})")
            all_tokens = extract_all_tokens_for_feature(
                indices=indices, values=values, feature_idx=fi,
                dataset_path=meta["dataset"], tokenizer=gen_tok,
                context_window=args.context_window, all_token_ids=all_token_ids,
                token_mean_act=None, alpha_filter=False,
                domain_mask=sap_label_mask,
            )
            if len(all_tokens) < args.min_active:
                print(f"  SKIPPED: n_active={len(all_tokens)} < {args.min_active}")
                continue
            top10_max = max((t["activation"] for t in all_tokens[:10]), default=0.0)
            threshold = 0.5 * top10_max
            desc_cache[fi] = {"threshold": threshold, "all_tokens": all_tokens, "quantiles": {}}
            for q_s in range(args.n_quantile_samples):
                q_seed = fi * 1000 + q_s
                h_seed = fi * 1000 + q_s + 500
                # Held-out set is fixed for this quantile sample (independent of the
                # explainer prompt), so sample it once outside the re-sample loop.
                held_out, held_out_idxs = _sample_held_out(
                    all_tokens, all_token_ids, gen_tok, indices, values,
                    fi, args.held_out_per_quantile, args.context_window, h_seed)

                t_desc = time.time()
                descriptions = []
                expl_rag_ctx_pre = None
                # Each description is generated independently. If SAP AI Core's content
                # filter blocks a prompt (content_filtered), we re-sample the quantile
                # tokens with a varied seed to build a genuinely different prompt and
                # retry — never append an empty description (empties corrupt metrics).
                MAX_RESAMPLE = 20
                for d_i in range(args.n_desc_samples):
                    got = False
                    for resample in range(MAX_RESAMPLE):
                        # Vary the seed per re-sample attempt to draw different tokens.
                        s_seed = q_seed + resample * 100000
                        expl_tokens = sample_from_quantiles(
                            all_tokens, args.explainer_tokens_per_quantile,
                            rng=np.random.default_rng(s_seed))
                        explainer_pool = expl_tokens
                        if use_explainer_rag:
                            top15_ctxs_expl = " ".join(
                                tok["context"].replace("\n", " ").strip()
                                for tok in sorted(all_tokens, key=lambda t: t["activation"],
                                                  reverse=True)[:15] if tok.get("context"))
                            if top15_ctxs_expl:
                                expl_chunks_pre = rag_client.retrieve(top15_ctxs_expl, top_k=args.rag_top_k)
                                if expl_chunks_pre:
                                    expl_rag_ctx_pre = "\n\n".join(
                                        f"[Passage {i+1}]: {c}" for i, c in enumerate(expl_chunks_pre))
                        try:
                            descriptions.append(bedrock_explainer.generate_text(
                                EXPLAINER_SYSTEM,
                                _build_explainer_user(fi, explainer_pool, rag_context=expl_rag_ctx_pre),
                                max_tokens=args.explainer_max_tokens,
                                temperature=args.temperature))
                            got = True
                            break
                        except ContentFilteredError:
                            print(f"  Q{q_s} desc{d_i}: content_filtered, re-sampling quantile tokens "
                                  f"(attempt {resample+1}/{MAX_RESAMPLE}, seed={s_seed})...", flush=True)
                            continue
                        except Exception as e:
                            print(f"  [BedrockExplainer] error: {e}", flush=True)
                            descriptions.append("")
                            got = True
                            break
                    if not got:
                        print(f"  Q{q_s} desc{d_i}: content_filtered on all {MAX_RESAMPLE} re-samples "
                              f"-> giving up, appending empty description", flush=True)
                        descriptions.append("")
                print(f"  Q{q_s}: {len(descriptions)} descriptions in {time.time()-t_desc:.1f}s")
                desc_cache[fi]["quantiles"][q_s] = {
                    "descriptions": descriptions,
                    "held_out": held_out,
                    "expl_rag_ctx_pre": expl_rag_ctx_pre,
                    "explainer_pool": explainer_pool,
                }
        print(f"\n[Phase1] Done in {(time.time()-t_phase1)/60:.1f}min. "
              f"Descriptions cached for {len(desc_cache)} features.\n")
        print(f"{'='*60}")
        print(f"PHASE 2: Judging with vLLM (no Bedrock calls)")
        print(f"{'='*60}\n")

    for feat_i, fi in enumerate(valid_features):
        t_feat = time.time()
        print(f"\n{'='*60}")
        print(f"Feature {fi}  ({feat_i+1}/{len(valid_features)})")
        print(f"{'='*60}")

        if args.dummy_descriptions or bedrock_explainer is not None:
            # Use pre-computed descriptions from Phase 1
            if fi not in desc_cache:
                print(f"  SKIPPED (no descriptions cached — likely n_active < min_active)")
                continue
            cached = desc_cache[fi]
            all_tokens = cached["all_tokens"]
            threshold = cached["threshold"]
        else:
            t = time.time()
            all_tokens = extract_all_tokens_for_feature(
                indices=indices, values=values, feature_idx=fi,
                dataset_path=meta["dataset"], tokenizer=gen_tok,
                context_window=args.context_window, all_token_ids=all_token_ids,
                token_mean_act=None, alpha_filter=False,
                domain_mask=sap_label_mask,
            )
            _t("token_extraction", t)

            if len(all_tokens) < args.min_active:
                print(f"  SKIPPED: n_active={len(all_tokens)} < {args.min_active}")
                continue

            top10_max = max((t["activation"] for t in all_tokens[:10]), default=0.0)
            threshold = 0.5 * top10_max

        for q_s in range(args.n_quantile_samples):
            t_qs = time.time()
            q_seed = fi * 1000 + q_s
            h_seed = fi * 1000 + q_s + 500

            if args.dummy_descriptions or bedrock_explainer is not None:
                # Load from Phase 1 cache (dummy or Bedrock)
                q_cached = cached["quantiles"][q_s]
                descriptions = q_cached["descriptions"]
                held_out = q_cached["held_out"]
                expl_rag_ctx_pre = q_cached["expl_rag_ctx_pre"]
                explainer_pool = q_cached.get("explainer_pool")
                desc_time = 0.0
            else:
                t = time.time()
                expl_tokens = sample_from_quantiles(
                    all_tokens, args.explainer_tokens_per_quantile,
                    rng=np.random.default_rng(q_seed))
                held_out, held_out_idxs = _sample_held_out(
                    all_tokens, all_token_ids, gen_tok, indices, values,
                    fi, args.held_out_per_quantile, args.context_window, h_seed)
                explainer_pool = expl_tokens
                _t(f"Q{q_s}/token_sampling", t)

                expl_rag_ctx_pre = None
                if use_explainer_rag:
                    t = time.time()
                    top15_ctxs_expl = " ".join(
                        tok["context"].replace("\n", " ").strip()
                        for tok in sorted(all_tokens, key=lambda t: t["activation"],
                                          reverse=True)[:15] if tok.get("context"))
                    if top15_ctxs_expl:
                        expl_chunks_pre = rag_client.retrieve(top15_ctxs_expl, top_k=args.rag_top_k)
                        if expl_chunks_pre:
                            expl_rag_ctx_pre = "\n\n".join(
                                f"[Passage {i+1}]: {c}" for i, c in enumerate(expl_chunks_pre))
                    _t(f"Q{q_s}/explainer_rag_retrieval", t)

                t_desc = time.time()
                desc_msgs = [_chat_prompt(EXPLAINER_SYSTEM,
                                          _build_explainer_user(fi, explainer_pool,
                                                                 rag_context=expl_rag_ctx_pre))
                             for _ in range(args.n_desc_samples)]
                desc_resps = explainer_router.chat_many(
                    desc_msgs,
                    max_tokens=args.explainer_max_tokens,
                    temperature=args.temperature,
                    chat_template_kwargs={"enable_thinking": False})
                descriptions = []
                for r in desc_resps:
                    if r is None or "error" in (r or {}):
                        print(f"  [explainer] empty/error response: {str(r)[:200]}")
                        descriptions.append("")
                    else:
                        descriptions.append(
                            r["choices"][0]["message"]["content"].strip())
                desc_time = time.time() - t_desc
            print(f"  Q{q_s}: {len(descriptions)} descriptions in {desc_time:.1f}s")

            t = time.time()
            qwen_rag_ctxs = {}
            valid_desc_indices = []
            if use_rag:
                top15_ctxs = " ".join(
                    tok["context"].replace("\n", " ").strip()
                    for tok in sorted(all_tokens, key=lambda t: t["activation"],
                                      reverse=True)[:15] if tok.get("context"))
                valid_desc_indices = [i for i, d in enumerate(descriptions) if d]
                judge_queries = [
                    descriptions[i] + (" " + top15_ctxs if top15_ctxs else "")
                    for i in valid_desc_indices
                ]
                all_chunks = rag_client.retrieve_batch(judge_queries, top_k=args.rag_top_k)
                for list_idx, orig_idx in enumerate(valid_desc_indices):
                    chunks = all_chunks[list_idx]
                    qwen_rag_ctxs[orig_idx] = (
                        "\n\n".join(f"[Passage {i+1}]: {c}" for i, c in enumerate(chunks))
                        if chunks else None)
            _t(f"Q{q_s}/judge_rag_retrieval (Qdrant, {len(valid_desc_indices)} queries)", t)

            for d_s, description in enumerate(descriptions):
                if not description and not args.dummy_descriptions:
                    continue

                judge_rag_ctx = qwen_rag_ctxs.get(d_s) if use_rag else None
                judge_suffix = args.judge_mode

                for m_s in range(args.n_metric_samples):
                    ctrl_seed = fi * 100000 + q_s * 1000 + d_s * 100 + m_s
                    print(f"  [D{d_s}/M{m_s}] CPT judge ...")

                    t = time.time()
                    cpt_sentences = _gen_sentences_endpoint(
                        cpt_router, description, args.gen_tokens,
                        args.generation_max_tokens, args.temperature,
                        rag_context=None)
                    _t(f"D{d_s}/M{m_s}/cpt_sentence_gen ({len(cpt_sentences)} sentences)", t)

                    cpt_metrics = _run_metrics(
                        "cpt", cpt_router, description, cpt_sentences,
                        held_out, threshold, fi,
                        gen_model, gen_tok, sae, sae_norm, meta,
                        hook_store, all_token_ids, device, args,
                        rag_context=None, ctrl_rng_seed=ctrl_seed)

                    print(f"  [D{d_s}/M{m_s}] {judge_suffix} judge ...")

                    t = time.time()
                    judge_sentences = _gen_sentences_endpoint(
                        qwen_router, description, args.gen_tokens,
                        args.generation_max_tokens, args.temperature,
                        rag_context=judge_rag_ctx)
                    _t(f"D{d_s}/M{m_s}/judge_sentence_gen ({len(judge_sentences)} sentences)", t)

                    judge_metrics = _run_metrics(
                        judge_suffix, qwen_router, description, judge_sentences,
                        held_out, threshold, fi,
                        gen_model, gen_tok, sae, sae_norm, meta,
                        hook_store, all_token_ids, device, args,
                        rag_context=judge_rag_ctx, ctrl_rng_seed=ctrl_seed + 1)

                    record = {
                        "feature_idx": fi,
                        "q_sample":    q_s,
                        "d_sample":    d_s,
                        "m_sample":    m_s,
                        "description": description,
                        "explainer_tokens": explainer_pool,
                        "held_out":    held_out,
                        "sentences_cpt":        cpt_sentences,
                        f"sentences_{judge_suffix}": judge_sentences,
                        **cpt_metrics,
                        **judge_metrics,
                    }
                    all_records.append(record)

            print(f"  [time] Q{q_s} total: {time.time()-t_qs:.1f}s")

        n_done = feat_i + 1
        elapsed = time.time() - t_loop
        per_feat = elapsed / n_done
        remaining = per_feat * (len(valid_features) - n_done)
        feat_total = time.time() - t_feat
        print(f"  [time] feature total: {feat_total:.1f}s")
        print(f"  -> {len(all_records)} records so far | "
              f"{elapsed/60:.1f}min elapsed | "
              f"~{remaining/60:.1f}min remaining")

        if n_done % 5 == 0:
            _write_output(args.output, all_records, setup_time,
                          time.time() - t_loop, args, meta, checkpoint=True)

    hook_handle.remove()
    total_time = time.time() - t_loop
    _write_output(args.output, all_records, setup_time, total_time, args, meta,
                  checkpoint=False)
    print(f"\nDone. {len(all_records)} records in {total_time/60:.1f}min.")
    print(f"Output: {args.output}")


def _write_output(path, records, setup_time, loop_time, args, meta, checkpoint=False):
    out = {
        "timestamp":    datetime.now().isoformat(),
        "checkpoint":   checkpoint,
        "n_records":    len(records),
        "setup_seconds": round(setup_time, 1),
        "loop_seconds":  round(loop_time, 1),
        "args": {
            "features":             args.features,
            "n_quantile_samples":   args.n_quantile_samples,
            "n_desc_samples":       args.n_desc_samples,
            "n_metric_samples":     args.n_metric_samples,
            "temperature":          args.temperature,
            "gen_tokens":           args.gen_tokens,
            "judge_mode":           args.judge_mode,
            "explainer_served_model": "anthropic--claude-4.7-opus (BedrockExplainer)" if args.sota else args.explainer_served_model,
            "cpt_served_model":     args.cpt_served_model,
            "qwen_served_model":    args.qwen_served_model,
            "qdrant_url":           args.qdrant_url,
            "qdrant_collection":    args.qdrant_collection,
            "rag_top_k":            args.rag_top_k,
            "dataset":              meta["dataset"],
            "layer":                meta["layer"],
        },
        "records": records,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
