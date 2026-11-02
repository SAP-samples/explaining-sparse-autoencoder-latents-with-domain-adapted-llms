"""
Compute bootstrap CIs for the stratified-50 feature results.

Reads 6 JSON files (BASE, BASE+RAG, CPT, CPT+RAG, SOTA, SOTA+RAG) and
produces up to 3 summary tables with different bootstrap strategies:

  --bootstrap within   : resample the 27 branching records within each feature,
                         then average across features  (measures within-feature noise)
  --bootstrap over     : resample features with replacement (N=50),
                         use raw mean of all their records  (measures between-feature variance)
  --bootstrap combined : resample features with replacement, AND within each
                         resampled feature resample its 27 records
                         (both sources of variance at once)

All three use 10 000 bootstrap iterations.  95 % CI = ±1.96 × std of bootstrap distribution.

Usage:
    python compute_bootstrap_ci.py \\
        --input-dir /shared/home/i777599/nova_model/SAE/results/tables/SAP/Gemma3-50-LLMaaJ-features \\
        --bootstrap within over combined \\
        --output-dir /shared/home/i777599/nova_model/SAE/results/tables/SAP/Gemma3-50-LLMaaJ-features
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── config ──────────────────────────────────────────────────────────────────
EXPLAINERS   = ["BASE", "BASE+RAG", "CPT", "CPT+RAG", "SOTA", "SOTA+RAG", "DUMMY"]
FILE_NAMES   = ["BASE", "BASE+RAG", "CPT", "CPT+RAG", "SOTA", "SOTA+RAG", "DUMMY"]

METRICS_CPT  = ["gen_acc_cpt",      "cosy_auc_cpt",      "cosy_mad_cpt",
                 "pred_acc_cpt",     "detection_auroc_cpt", "fuzzing_acc_cpt"]
METRICS_RAG  = ["gen_acc_base_rag", "cosy_auc_base_rag", "cosy_mad_base_rag",
                 "pred_acc_base_rag","detection_auroc_base_rag","fuzzing_acc_base_rag"]

DISPLAY_COLS = ["gen_acc", "cosy_auc", "cosy_mad", "pred_acc", "det_auroc", "fuzzing"]

N_BOOTSTRAP  = 10_000
SEED         = 42


# ── data loading ────────────────────────────────────────────────────────────

def load_data(input_dir: Path) -> dict:
    """
    Returns nested dict:
        data[explainer][feature_idx] = list of record dicts
    """
    data = {}
    for name in FILE_NAMES:
        path = input_dir / f"{name}.json"
        with open(path) as f:
            obj = json.load(f)
        by_feature = defaultdict(list)
        for rec in obj["records"]:
            by_feature[rec["feature_idx"]].append(rec)
        data[name] = dict(by_feature)
    return data


# ── bootstrap implementations ────────────────────────────────────────────────

def _feature_means(records_for_feature: list, metrics: list) -> np.ndarray:
    """Mean of each metric over the branching records for one feature. None → nan."""
    arr = np.array([[r.get(m) if r.get(m) is not None else np.nan
                     for m in metrics]
                    for r in records_for_feature], dtype=np.float64)
    return np.nanmean(arr, axis=0)


def _build_matrix(feature_records: dict, metrics: list) -> np.ndarray:
    """
    Returns array of shape (n_features, n_records, n_metrics).
    Assumes all features have the same number of records.
    """
    features = list(feature_records.keys())
    rows = []
    for fid in features:
        recs = feature_records[fid]
        feat_arr = np.array([[r.get(m) if r.get(m) is not None else np.nan
                               for m in metrics] for r in recs], dtype=np.float64)
        rows.append(feat_arr)
    return np.stack(rows, axis=0)  # (F, R, M)


def bootstrap_within(feature_records: dict, metrics: list,
                     n_iter: int = N_BOOTSTRAP, rng=None) -> tuple:
    """Resample records within each feature; average over features."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    mat = _build_matrix(feature_records, metrics)  # (F, R, M)
    F, R, M = mat.shape
    # sample indices: (n_iter, F, R)
    idx = rng.integers(0, R, size=(n_iter, F, R))
    # gather: (n_iter, F, R, M)
    sampled = mat[np.arange(F)[None, :, None], idx]  # (n_iter, F, R, M)
    boot_means = np.nanmean(sampled, axis=2).mean(axis=1)  # (n_iter, M)
    grand_mean = boot_means.mean(axis=0)
    ci_half    = (np.percentile(boot_means, 97.5, axis=0) - np.percentile(boot_means, 2.5, axis=0)) / 2
    return grand_mean, ci_half


def bootstrap_over(feature_records: dict, metrics: list,
                   n_iter: int = N_BOOTSTRAP, rng=None) -> tuple:
    """Resample features with replacement; use full mean of each feature's records."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    mat = _build_matrix(feature_records, metrics)  # (F, R, M)
    F = mat.shape[0]
    feat_means = np.nanmean(mat, axis=1)  # (F, M)
    # sample feature indices: (n_iter, F)
    idx = rng.integers(0, F, size=(n_iter, F))
    boot_means = feat_means[idx].mean(axis=1)  # (n_iter, M)
    grand_mean = boot_means.mean(axis=0)
    ci_half    = (np.percentile(boot_means, 97.5, axis=0) - np.percentile(boot_means, 2.5, axis=0)) / 2
    return grand_mean, ci_half


def bootstrap_combined(feature_records: dict, metrics: list,
                       n_iter: int = N_BOOTSTRAP, rng=None) -> tuple:
    """Resample features with replacement AND resample records within each."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    mat = _build_matrix(feature_records, metrics)  # (F, R, M)
    F, R, M = mat.shape
    # sample feature indices: (n_iter, F)
    feat_idx = rng.integers(0, F, size=(n_iter, F))
    # sample record indices: (n_iter, F, R)
    rec_idx = rng.integers(0, R, size=(n_iter, F, R))
    # gather: for each (iter, feat_slot, record_slot) → mat[feat_idx[i,j], rec_idx[i,j,k], :]
    # mat[feat_idx[:,:,None], rec_idx] → shape (n_iter, F, R, M)
    sampled = mat[feat_idx[:, :, None], rec_idx, :]  # (n_iter, F, R, M)
    boot_means = np.nanmean(sampled, axis=2).mean(axis=1)  # (n_iter, M)
    grand_mean = boot_means.mean(axis=0)
    ci_half    = (np.percentile(boot_means, 97.5, axis=0) - np.percentile(boot_means, 2.5, axis=0)) / 2
    return grand_mean, ci_half


# ── table formatting ─────────────────────────────────────────────────────────

def _row(label, means, cis, col_w=18):
    cells = [f"{m:.4f} ± {c:.4f}".center(col_w) for m, c in zip(means, cis)]
    return "| " + label.center(8) + " | " + " | ".join(cells) + " |"


def _separator(col_w=18, n_cols=6):
    return "+" + "-" * 10 + "+" + ("-" * (col_w + 2) + "+") * n_cols


def _header(col_w=18):
    cols = ["gen_acc", "cosy_auc", "cosy_mad", "pred_acc", "det_auroc", "fuzzing"]
    return ("| " + "File".center(8) + " | "
            + " | ".join(c.center(col_w) for c in cols) + " |")


def build_table(results: dict) -> str:
    """
    results: {explainer: (means_array, ci_array)}  — n_metrics values each
    """
    sep = _separator()
    lines = [sep, _header(), sep]
    for expl in EXPLAINERS:
        means, cis = results[expl]
        lines.append(_row(expl, means, cis))
        lines.append(sep)
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  type=str,
                        default="/shared/home/i777599/nova_model/SAE/results/tables/SAP/Gemma3-50-LLMaaJ-features")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Defaults to --input-dir")
    parser.add_argument("--bootstrap",  nargs="+",
                        choices=["within", "over", "combined"],
                        default=["within", "over", "combined"],
                        help="Which bootstrap variants to compute")
    parser.add_argument("--n-iter",     type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    print("Loading JSON files...")
    data = load_data(input_dir)

    bootstrap_fns = {
        "within":   bootstrap_within,
        "over":     bootstrap_over,
        "combined": bootstrap_combined,
    }

    for variant in args.bootstrap:
        fn = bootstrap_fns[variant]
        print(f"\nRunning bootstrap '{variant}' ({args.n_iter} iterations)...")

        sections = {}

        for judge_idx, (judge_label, metrics) in enumerate([("CPT Judge",      METRICS_CPT),
                                                             ("BASE+RAG Judge", METRICS_RAG)]):
            results = {}
            for expl in EXPLAINERS:
                # fresh rng per (variant, judge, explainer) for reproducibility
                local_rng = np.random.default_rng(SEED + hash((variant, judge_idx, expl)) % 2**31)
                means, cis = fn(data[expl], metrics, n_iter=args.n_iter, rng=local_rng)
                results[expl] = (means, cis)
            sections[judge_label] = build_table(results)

        # assemble output text
        lines = []
        for judge_label, table in sections.items():
            lines.append(f"\n=== {judge_label} ===\n")
            lines.append(table)
            lines.append("")
        output_text = "\n".join(lines)

        out_path = output_dir / f"final50_stratified_summary_bootstrap_{variant}.txt"
        out_path.write_text(output_text)
        print(f"  Written: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
