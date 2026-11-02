"""
MMLU evaluation: BASE (g524xlarge-5:18003) vs CPT (g524xlarge-6:18002)
Both are Gemma-3-12B variants served via vLLM.
Uses 5-shot MMLU, parallel async requests, outputs JSON + TXT.
"""

import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from openai import AsyncOpenAI

# ── config ───────────────────────────────────────────────────────────────────
BASE_URL   = "http://gpu-dy-g524xlarge-5:18003/v1"
CPT_URL    = "http://gpu-dy-g524xlarge-6:18002/v1"
BASE_MODEL = "gemma-3-12b-it"
CPT_MODEL  = "cpt-v6-iftv2"

N_SAMPLES    = None     # None = all questions
N_FEWSHOT    = 0
CONCURRENCY  = 32
SEED         = 42
TEMPERATURE  = 0.0
MAX_TOKENS   = 5        # only need A/B/C/D

OUTPUT_DIR   = Path("/shared/home/i777599/nova_model/SAE/results/MMLU_results")

CHOICES = ["A", "B", "C", "D"]

# ── few-shot helpers ──────────────────────────────────────────────────────────

def format_question(row, include_answer=False):
    q = row["question"].strip()
    opts = "\n".join(f"{c}. {row['choices'][i]}" for i, c in enumerate(CHOICES))
    prompt = f"Question: {q}\n{opts}\nAnswer:"
    if include_answer:
        prompt += f" {CHOICES[row['answer']]}"
    return prompt


def build_prompt(fewshot_rows, test_row):
    parts = ["The following are multiple choice questions. Answer with only the letter (A/B/C/D).\n"]
    for r in fewshot_rows:
        parts.append(format_question(r, include_answer=True))
    parts.append(format_question(test_row, include_answer=False))
    return "\n\n".join(parts)


def extract_answer(text: str):
    text = text.strip()
    for c in CHOICES:
        if text.upper().startswith(c):
            return c
    # fallback: first letter found
    for ch in text.upper():
        if ch in CHOICES:
            return ch
    return None


# ── async eval ───────────────────────────────────────────────────────────────

async def eval_one(client, model, prompt, correct):
    try:
        resp = await asyncio.wait_for(
            client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ),
            timeout=120.0,
        )
        raw = resp.choices[0].text
        pred = extract_answer(raw)
        return pred, raw.strip()
    except asyncio.TimeoutError:
        print(f"  [warn] timeout for {model}")
        return None, None
    except Exception as e:
        print(f"  [warn] request failed: {e}")
        return None, None


async def eval_model(url, model, prompts_and_answers, label, jsonl_file=None, jsonl_lock=None, rows=None):
    client = AsyncOpenAI(api_key="x", base_url=url, timeout=180.0)
    sem = asyncio.Semaphore(CONCURRENCY)
    done_count = 0

    async def bounded(i, prompt, correct):
        nonlocal done_count
        async with sem:
            pred, raw = await eval_one(client, model, prompt, correct)
            done_count += 1
            if done_count % 50 == 0:
                print(f"  [{label}] {done_count}/{len(prompts_and_answers)} done...")
            # write to JSONL immediately
            if jsonl_file and jsonl_lock:
                row = rows[i] if rows else {}
                entry = {
                    "idx": i,
                    "subject": row.get("subject", ""),
                    "question": row.get("question", ""),
                    "correct": correct,
                    "model": label,
                    "pred": pred,
                    "raw": raw,
                    "correct_bool": pred == correct if pred else None,
                }
                async with jsonl_lock:
                    jsonl_file.write(json.dumps(entry) + "\n")
                    jsonl_file.flush()
            return pred, raw

    tasks = [bounded(i, p, a) for i, (p, a) in enumerate(prompts_and_answers)]
    t0 = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    correct_list = [prompts_and_answers[i][1] for i in range(len(results))]
    valid = [(pred, raw, correct_list[i]) for i, (pred, raw) in enumerate(results) if pred is not None]
    acc = sum(1 for pred, raw, correct in valid if pred == correct) / len(valid) if valid else 0.0
    print(f"  [{label}] acc={acc:.4f}  ({sum(1 for p,r,c in valid if p==c)}/{len(valid)})  {elapsed:.1f}s")
    return acc, results


# ── main ─────────────────────────────────────────────────────────────────────

async def main():
    rng = random.Random(SEED)

    print("Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="validation")

    # group by subject for few-shot sampling
    dev_by_subject = {}
    for row in dev_ds:
        dev_by_subject.setdefault(row["subject"], []).append(row)

    test_rows = list(ds)
    rng.shuffle(test_rows)
    if N_SAMPLES is not None:
        test_rows = test_rows[:N_SAMPLES]

    print(f"  {len(test_rows)} test questions across {len(set(r['subject'] for r in test_rows))} subjects")

    # build prompts
    prompts_and_answers = []
    for row in test_rows:
        subject = row["subject"]
        fewshot = dev_by_subject.get(subject, [])
        if len(fewshot) >= N_FEWSHOT:
            shots = rng.sample(fewshot, N_FEWSHOT)
        else:
            shots = fewshot  # use whatever is available
        prompt = build_prompt(shots, row)
        correct = CHOICES[row["answer"]]
        prompts_and_answers.append((prompt, correct))

    print(f"\nEvaluating BASE @ {BASE_URL}  +  CPT @ {CPT_URL}  [parallel]")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = OUTPUT_DIR / f"mmlu_base_vs_cpt_{timestamp}.jsonl"
    jsonl_lock = asyncio.Lock()
    with open(jsonl_path, "w") as jsonl_file:
        (base_acc, base_results), (cpt_acc, cpt_results) = await asyncio.gather(
            eval_model(BASE_URL, BASE_MODEL, prompts_and_answers, "BASE", jsonl_file, jsonl_lock, test_rows),
            eval_model(CPT_URL,  CPT_MODEL,  prompts_and_answers, "CPT",  jsonl_file, jsonl_lock, test_rows),
        )
    print(f"JSONL saved: {jsonl_path}")

    # ── per-question + per-subject breakdown ─────────────────────────────────
    questions_log = []
    subject_stats = {}
    for i, row in enumerate(test_rows):
        s = row["subject"]
        correct = CHOICES[row["answer"]]
        base_pred, base_raw = base_results[i] if i < len(base_results) else (None, None)
        cpt_pred,  cpt_raw  = cpt_results[i]  if i < len(cpt_results)  else (None, None)
        questions_log.append({
            "idx": i,
            "subject": s,
            "question": row["question"],
            "choices": {c: row["choices"][j] for j, c in enumerate(CHOICES)},
            "correct": correct,
            "base_pred": base_pred,
            "base_raw": base_raw,
            "base_correct": base_pred == correct if base_pred else None,
            "cpt_pred": cpt_pred,
            "cpt_raw": cpt_raw,
            "cpt_correct": cpt_pred == correct if cpt_pred else None,
        })
        if s not in subject_stats:
            subject_stats[s] = {"base": [], "cpt": []}
        if base_pred is not None:
            subject_stats[s]["base"].append(int(base_pred == correct))
        if cpt_pred is not None:
            subject_stats[s]["cpt"].append(int(cpt_pred == correct))

    per_subject = {}
    for s, v in subject_stats.items():
        per_subject[s] = {
            "base_acc": sum(v["base"]) / len(v["base"]) if v["base"] else None,
            "cpt_acc":  sum(v["cpt"])  / len(v["cpt"])  if v["cpt"]  else None,
            "n": len(v["base"]),
        }

    # ── save final JSON + TXT ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "n_questions": len(test_rows),
        "n_fewshot": N_FEWSHOT,
        "seed": SEED,
        "base_endpoint": BASE_URL,
        "cpt_endpoint": CPT_URL,
        "base_model": BASE_MODEL,
        "cpt_model": CPT_MODEL,
        "base_acc": base_acc,
        "cpt_acc": cpt_acc,
        "delta_cpt_minus_base": cpt_acc - base_acc,
        "per_subject": per_subject,
        "questions": questions_log,
    }

    json_path = OUTPUT_DIR / f"mmlu_base_vs_cpt_{timestamp}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\nJSON saved: {json_path}")

    # ── save TXT summary ──────────────────────────────────────────────────────
    lines = [
        "=" * 60,
        f"MMLU: BASE vs CPT — {timestamp}",
        "=" * 60,
        f"  Questions : {len(test_rows)}  (seed={SEED}, {N_FEWSHOT}-shot)",
        f"  BASE ({BASE_MODEL}) : {base_acc:.4f}",
        f"  CPT  ({CPT_MODEL})  : {cpt_acc:.4f}",
        f"  Delta (CPT-BASE)   : {cpt_acc - base_acc:+.4f}",
        "",
        f"  {'Subject':<45} {'BASE':>6}  {'CPT':>6}  {'Delta':>7}  {'N':>4}",
        "-" * 75,
    ]
    for s, v in sorted(per_subject.items(), key=lambda x: -(x[1]["cpt_acc"] or 0) + (x[1]["base_acc"] or 0)):
        ba = v["base_acc"]
        ca = v["cpt_acc"]
        if ba is None or ca is None:
            continue
        lines.append(f"  {s:<45} {ba:>6.3f}  {ca:>6.3f}  {ca-ba:>+7.3f}  {v['n']:>4}")
    lines += ["", "=" * 60]

    txt_path = OUTPUT_DIR / f"mmlu_base_vs_cpt_{timestamp}.txt"
    txt_path.write_text("\n".join(lines))
    print(f"TXT  saved: {txt_path}")

    print("\n" + "\n".join(lines[1:8]))


if __name__ == "__main__":
    asyncio.run(main())
