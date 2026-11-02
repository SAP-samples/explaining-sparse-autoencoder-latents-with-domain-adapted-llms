"""
CrossCoder Feature Viewer + Pipeline Manager
Flask app — feature browser + end-to-end run management.
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

# Add pipeline dir to path for run_manager import
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
import run_manager as rm

app = Flask(__name__)

_STORAGE_ROOT = Path(os.environ.get("CC_STORAGE_ROOT", Path.home() / "Webapp-Storage"))
_RUN_ID_RE = re.compile(r'^run_\d{8}_\d{6}_[0-9a-f]{6}$')
_SETTINGS_PATH = _STORAGE_ROOT / "settings.json"

_DEFAULT_SETTINGS = {"label_a": "Model A", "label_b": "Model B", "norm_thresh_lo": 0.1, "norm_thresh_hi": 0.9, "bookmarks": {}}
_settings: dict = {}


def _load_settings() -> dict:
    global _settings
    if _SETTINGS_PATH.exists():
        try:
            _settings = json.loads(_SETTINGS_PATH.read_text())
        except Exception:
            _settings = {}
    merged = {**_DEFAULT_SETTINGS, **_settings}
    _settings = merged
    return merged


def _save_settings(data: dict):
    global _settings
    _settings.update(data)
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json.dumps(_settings, indent=2))

# Rate limiting for /api/runs/create
_create_run_times: list = []
_create_run_lock = threading.Lock()
_CSV_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Active CSV state (can be switched per-run)
# ---------------------------------------------------------------------------
_DEFAULT_CSV = os.environ.get(
    "CC_CSV",
    str(Path(__file__).parent.parent / "data" / "llmaaj_results" / "crosscoder_features_adjusted_thresholds_20261012.csv")
)

INDEX_COLS = [
    "feature_idx", "domain_label", "judge_label", "judge_score",
    "norm_flag", "delta_norm", "n_active",
    "judge_domain_purity", "judge_lang_purity",
    "judge_content_richness", "judge_specificity",
]

_csv_path   = _DEFAULT_CSV
_rows_full  = []
_index      = []
_fidx_map: dict = {}  # feature_idx (int) → full row dict for O(1) lookup

# ---------------------------------------------------------------------------
# CrossCoder activation context lookup state
# ---------------------------------------------------------------------------
_cc_indices     = None   # np.memmap (N, topk) int32
_cc_values      = None   # np.memmap (N, topk) float32
_token_map      = None   # np.array  (N, 2)   int32: [chunk_idx, tok_pos]
_token_char_offsets = None  # np.array (N, 2) int32: [char_start, char_end] in chunk text
_chunk_texts:   list = []
_chunk_domains: list = []
_cc_meta:       dict = {}
_cc_tokenizer         = None  # cached tokenizer for context extraction (fallback only)

_KNOWN_CC_META = ""  # cleared — old run (layer25, f36864) not relevant; new runs pending


def load_cc_acts(meta_path: str):
    global _cc_indices, _cc_values, _token_map, _token_char_offsets, _chunk_texts, _chunk_domains, _cc_meta, _cc_tokenizer
    import torch
    mp = Path(meta_path)
    if not mp.exists():
        print(f"[cc_acts] meta not found: {mp}", flush=True)
        return False

    meta = torch.load(str(mp), map_location="cpu", weights_only=False)
    base = mp.parent / mp.stem.replace("_meta", "")

    indices_path       = Path(meta["indices_path"])
    values_path        = Path(meta["values_path"])
    token_map_path     = Path(str(base) + "_token_map.npy")
    chunk_texts_path   = Path(str(base) + "_chunk_texts.npy")
    chunk_domains_path = Path(str(base) + "_chunk_domains.npy")
    char_offsets_path  = Path(str(base) + "_token_char_offsets.npy")

    required = [indices_path, values_path, token_map_path, chunk_texts_path, chunk_domains_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"[cc_acts] missing files: {[str(p) for p in missing]}", flush=True)
        if token_map_path in missing:
            print(f"[cc_acts] Run build_token_map.py first to generate token_map files.", flush=True)
        return False

    n  = meta["n_tokens"]
    k  = meta["topk"]
    print(f"[cc_acts] Loading {n:,} tokens, topk={k} ...", flush=True)
    _cc_indices = np.memmap(str(indices_path),  dtype="int32",   mode="r", shape=(n, k))
    _cc_values  = np.memmap(str(values_path),   dtype="float32", mode="r", shape=(n, k))
    _token_map  = np.load(str(token_map_path))
    _chunk_texts   = list(np.load(str(chunk_texts_path),  allow_pickle=True))
    _chunk_domains = list(np.load(str(chunk_domains_path), allow_pickle=True))
    _cc_meta = meta

    # Load char offsets if available (built by updated build_token_map.py)
    if char_offsets_path.exists():
        _token_char_offsets = np.load(str(char_offsets_path))
        print(f"[cc_acts] Char offsets loaded ({_token_char_offsets.shape[0]:,} tokens) — exact highlighting enabled.", flush=True)
    else:
        _token_char_offsets = None
        print(f"[cc_acts] No char offsets file found — falling back to tokenizer re-run for context.", flush=True)
        # Load tokenizer as fallback
        tok_id = meta.get("model_base", "google/gemma-2-2b")
        print(f"[cc_acts] Loading tokenizer {tok_id} ...", flush=True)
        from transformers import AutoTokenizer
        _cc_tokenizer = AutoTokenizer.from_pretrained(tok_id)
        if _cc_tokenizer.pad_token is None:
            _cc_tokenizer.pad_token = _cc_tokenizer.eos_token

    print(f"[cc_acts] Ready: {n:,} tokens, {len(_chunk_texts):,} chunks.", flush=True)
    return True


def load_csv(path: str = None):
    global _csv_path, _rows_full, _index, _fidx_map
    p = path or _csv_path
    if not Path(p).exists():
        print(f"[warn] CSV not found: {p}", flush=True)
        with _CSV_LOCK:
            _rows_full = []
            _index = []
            _fidx_map = {}
        return
    print(f"Loading CSV: {p} ...", flush=True)
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Normalize legacy column names to current schema
    _COL_ALIASES = {
        "lang_label":        "domain_label",
        "norm_base":         "norm_a",
        "norm_swallow":      "norm_b",
        "freq_base":         "freq_a",
        "freq_swallow":      "freq_b",
        "judge_lang_purity": "judge_domain_purity",
    }
    if rows and any(alias in rows[0] for alias in _COL_ALIASES):
        normalized = []
        for r in rows:
            nr = dict(r)
            for old, new in _COL_ALIASES.items():
                if old in nr and new not in nr:
                    nr[new] = nr.pop(old)
            normalized.append(nr)
        rows = normalized
        print(f"  [compat] Normalized legacy column names.", flush=True)

    idx = [{k: r.get(k, "") for k in INDEX_COLS} for r in rows]
    fidx_map = {}
    for r in rows:
        try:
            fidx_map[int(r["feature_idx"])] = r
        except (ValueError, KeyError):
            pass
    with _CSV_LOCK:
        _rows_full = rows
        _index = idx
        _fidx_map = fidx_map
        _csv_path = p
    print(f"  Loaded {len(_rows_full):,} features.", flush=True)


def _validate_run_id(run_id: str):
    """Return (None, None) if valid, or (response, code) if invalid."""
    if not _RUN_ID_RE.match(run_id):
        return jsonify({"error": "invalid run_id"}), 400
    return None, None


def _safe_send_file(path: str, download_name: str):
    """Serve a file only if it resolves inside STORAGE_ROOT or the repo."""
    resolved = Path(path).resolve()
    allowed_roots = [Path(str(_STORAGE_ROOT)).resolve(), Path(__file__).parent.parent.resolve()]
    if not any(str(resolved).startswith(str(r)) for r in allowed_roots):
        return jsonify({"error": "access denied"}), 403
    if not resolved.exists():
        return jsonify({"error": "file not found"}), 404
    return send_file(str(resolved), as_attachment=True,
                     download_name=download_name, mimetype="text/csv")


_load_settings()
load_csv()
if _KNOWN_CC_META and Path(_KNOWN_CC_META).exists():
    load_cc_acts(_KNOWN_CC_META)

# ---------------------------------------------------------------------------
# Feature-browser API
# ---------------------------------------------------------------------------

def _safe_int(v, default=0):
    try: return int(v)
    except (ValueError, TypeError): return default


@app.route("/api/index")
def api_index():
    q      = request.args.get("q", "").lower()
    lang   = request.args.get("lang", "")
    nflag  = request.args.get("norm_flag", "")
    sort   = request.args.get("sort", "delta_norm")
    order  = request.args.get("order", "desc")
    try:
        page         = max(0, int(request.args.get("page", 0)))
        limit        = min(max(1, int(request.args.get("limit", 50))), 500)
        n_active_min = max(0, int(request.args.get("n_active_min", 0)))
    except (ValueError, TypeError):
        return jsonify({"error": "invalid page/limit"}), 400

    rows = _index
    if lang:
        rows = [r for r in rows if r["domain_label"] == lang]
    if nflag:
        # Apply thresholds from settings live — not the frozen norm_flag from CSV
        lo = float(_settings.get("norm_thresh_lo", 0.1))
        hi = float(_settings.get("norm_thresh_hi", 0.9))
        def _live_norm_flag(dn_str):
            try:
                dn = float(dn_str)
            except (ValueError, TypeError):
                return "BOTH"
            if dn <= lo: return "A"
            if dn >= hi: return "B"
            return "BOTH"
        rows = [r for r in rows if _live_norm_flag(r.get("delta_norm", "")) == nflag]
    if q:
        rows = [r for r in rows if q in r.get("judge_label", "").lower() or q in r.get("feature_idx", "")]
    if n_active_min > 0:
        rows = [r for r in rows if _safe_int(r.get("n_active", 0)) >= n_active_min]

    reverse = (order == "desc")

    def sort_key(r):
        # judge_domain_purity: fall back to judge_lang_purity for legacy CSVs
        if sort == "judge_domain_purity":
            v = r.get("judge_domain_purity") or r.get("judge_lang_purity", "")
        else:
            v = r.get(sort, "")
        try:
            return float(v)
        except (ValueError, TypeError):
            return str(v)

    rows = sorted(rows, key=sort_key, reverse=reverse)
    total = len(rows)
    paged = rows[page * limit: (page + 1) * limit]
    return jsonify({"total": total, "rows": paged, "page": page, "limit": limit})


@app.route("/api/feature/<int:fidx>")
def api_feature(fidx):
    r = _fidx_map.get(fidx)
    if r is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(r)


@app.route("/api/feature/<int:fidx>/contexts")
def api_feature_contexts(fidx):
    if _cc_indices is None:
        return jsonify({"error": "no cc_acts loaded", "positions": []}), 200
    n = min(int(request.args.get("n", 20)), 50)

    # Find all global token positions where this feature fires
    row_mask = np.any(_cc_indices == fidx, axis=1)
    positions = np.where(row_mask)[0]

    results = []
    for pos in positions:
        row = _cc_indices[pos]           # (topk,)
        col_hits = np.where(row == fidx)[0]
        if len(col_hits) == 0:
            continue
        activation = float(_cc_values[pos, col_hits[0]])

        chunk_idx, tok_pos = int(_token_map[pos, 0]), int(_token_map[pos, 1])
        if chunk_idx >= len(_chunk_texts):
            continue
        text = _chunk_texts[chunk_idx]
        domain = int(_chunk_domains[chunk_idx]) if chunk_idx < len(_chunk_domains) else 0

        # Get char offset and token string
        token_str = ""
        context   = text[:200]  # fallback
        char_start_in_ctx = -1  # byte offset of target token within the returned context string

        if _token_char_offsets is not None:
            # Fast path: use pre-computed char offsets — no tokenizer call needed
            char_start = int(_token_char_offsets[pos, 0])
            char_end   = int(_token_char_offsets[pos, 1])
            token_str  = text[char_start:char_end]
            ctx_lo = max(0, char_start - 100)
            ctx_hi = min(len(text), char_end + 100)
            context = text[ctx_lo:ctx_hi]
            char_start_in_ctx = char_start - ctx_lo
        else:
            # Fallback: re-tokenize chunk to find char offset
            try:
                enc = _cc_tokenizer(text, truncation=True, max_length=2048,
                                    return_offsets_mapping=True, add_special_tokens=True)
                offsets = enc["offset_mapping"]
                token_ids = enc["input_ids"]
                if tok_pos < len(token_ids):
                    token_str = _cc_tokenizer.convert_ids_to_tokens([token_ids[tok_pos]])[0]
                if tok_pos < len(offsets):
                    char_start, char_end = offsets[tok_pos]
                    ctx_lo = max(0, char_start - 100)
                    ctx_hi = min(len(text), char_end + 100)
                    context = text[ctx_lo:ctx_hi]
                    char_start_in_ctx = char_start - ctx_lo
            except Exception:
                pass

        results.append({
            "token":            token_str,
            "activation":       round(activation, 4),
            "context":          context,
            "char_start_in_ctx": char_start_in_ctx,
            "token_len":        len(token_str),
            "chunk_idx":        chunk_idx,
            "tok_pos":          tok_pos,
            "domain":           domain,
        })

    # Sort by activation descending, return top-n
    results.sort(key=lambda x: x["activation"], reverse=True)
    return jsonify({"feature_idx": fidx, "total": len(results), "positions": results[:n]})


@app.route("/api/load_cc_acts", methods=["POST"])
def api_load_cc_acts():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json(silent=True) or {}
    meta_path = str(data.get("meta_path", "")).strip()
    if not meta_path:
        return jsonify({"error": "meta_path required"}), 400
    if not meta_path.startswith("/shared/home/i777599/"):
        return jsonify({"error": "access denied"}), 403
    ok = load_cc_acts(meta_path)
    if not ok:
        return jsonify({"error": "failed to load cc_acts — check server logs"}), 500
    return jsonify({"ok": True, "n_tokens": int(_cc_meta.get("n_tokens", 0))})


@app.route("/api/delta_norm_hist")
def api_delta_norm_hist():
    import numpy as np
    lo = float(request.args.get("lo", 0.1))
    hi = float(request.args.get("hi", 0.9))
    n_bins = 40
    vals = []
    for r in _index:
        try:
            vals.append(float(r["delta_norm"]))
        except (ValueError, TypeError):
            pass
    arr = np.array(vals) if vals else np.array([0.5])
    counts, edges = np.histogram(arr, bins=n_bins, range=(0.0, 1.0))
    bins = [{"lo": float(edges[i]), "hi": float(edges[i+1]), "n": int(counts[i])} for i in range(n_bins)]
    return jsonify({
        "bins": bins, "lo": lo, "hi": hi,
        "n_base":    int((arr < lo).sum()),
        "n_swallow": int((arr > hi).sum()),
        "n_both":    int(((arr >= lo) & (arr <= hi)).sum()),
        "total": len(arr), "mean": float(arr.mean()), "std": float(arr.std()),
    })


@app.route("/api/stats")
def api_stats():
    lang_counts = Counter(r["domain_label"] for r in _index)
    # Live norm counts using current settings thresholds
    lo = float(_settings.get("norm_thresh_lo", 0.1))
    hi = float(_settings.get("norm_thresh_hi", 0.9))
    norm_counts: Counter = Counter()
    for r in _index:
        try:
            dn = float(r["delta_norm"])
        except (ValueError, TypeError):
            norm_counts["BOTH"] += 1
            continue
        if dn <= lo:   norm_counts["A"] += 1
        elif dn >= hi: norm_counts["B"] += 1
        else:          norm_counts["BOTH"] += 1
    scores = [float(r["judge_score"]) for r in _index if r["judge_score"] not in ("", None, "-99")]
    return jsonify({
        "total": len(_index),
        "lang": dict(lang_counts),
        "norm_flag": dict(norm_counts),
        "score_mean": round(sum(scores)/len(scores), 2) if scores else 0,
        "score_max":  max(scores) if scores else 0,
        "csv_path": Path(_csv_path).name if _csv_path else "",
        "label_a": _settings.get("label_a", "Model A"),
        "label_b": _settings.get("label_b", "Model B"),
    })


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify(_settings)
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json(silent=True) or {}
    allowed_str = {"label_a", "label_b"}
    allowed_float = {"norm_thresh_lo", "norm_thresh_hi"}
    updates = {k: str(v)[:64] for k, v in data.items() if k in allowed_str}
    for k in allowed_float:
        if k in data:
            try:
                updates[k] = float(data[k])
            except (TypeError, ValueError):
                pass
    if "bookmarks" in data and isinstance(data["bookmarks"], dict):
        updates["bookmarks"] = {str(k): str(v)[:128] for k, v in data["bookmarks"].items()}
    if not updates:
        return jsonify({"error": "No valid fields"}), 400
    _save_settings(updates)
    return jsonify({"ok": True, **_settings})

# ---------------------------------------------------------------------------
# Run-management API
# ---------------------------------------------------------------------------

@app.route("/api/runs")
def api_runs():
    return jsonify(rm.api_list_runs())


@app.route("/api/runs/<run_id>")
def api_run(run_id):
    err, code = _validate_run_id(run_id)
    if err:
        return err, code
    try:
        return jsonify(rm.api_get_run(run_id))
    except FileNotFoundError:
        return jsonify({"error": "run not found"}), 404


@app.route("/api/download_csv")
def api_download_csv():
    """Download the currently active CSV file."""
    if not _csv_path or not Path(_csv_path).exists():
        return jsonify({"error": "No CSV loaded"}), 404
    return _safe_send_file(_csv_path, Path(_csv_path).name)


@app.route("/api/runs/<run_id>/download_csv")
def api_download_run_csv(run_id):
    """Download the CSV for a specific run."""
    err, code = _validate_run_id(run_id)
    if err:
        return err, code
    try:
        artifacts = rm.load_artifacts(run_id)
        csv_p = artifacts.get("llmaaj_csv", "")
        if not csv_p:
            return jsonify({"error": "CSV artifact not registered for this run"}), 404
        return _safe_send_file(csv_p, Path(csv_p).name)
    except FileNotFoundError:
        return jsonify({"error": "run not found"}), 404
    except Exception as e:
        return jsonify({"error": "internal error"}), 500


@app.route("/api/runs/<run_id>/load_csv", methods=["POST"])
def api_load_run_csv(run_id):
    """Switch the feature browser to the CSV of a finished run."""
    err, code = _validate_run_id(run_id)
    if err:
        return err, code
    try:
        artifacts = rm.load_artifacts(run_id)
        csv_p = artifacts.get("llmaaj_csv", "")
        if not csv_p or not Path(csv_p).exists():
            return jsonify({"error": "CSV not found for this run"}), 404
        load_csv(csv_p)
        return jsonify({"ok": True, "csv": Path(csv_p).name, "n_features": len(_rows_full)})
    except FileNotFoundError:
        return jsonify({"error": "run not found"}), 404
    except Exception as e:
        return jsonify({"error": "internal error"}), 500


@app.route("/api/runs/<run_id>/fve")
def api_run_fve(run_id):
    """Return parsed FVE + dead neuron metrics for a run, or null if not available."""
    err, code = _validate_run_id(run_id)
    if err:
        return err, code
    try:
        artifacts = rm.load_artifacts(run_id)
        fve_path = artifacts.get("fve_json", "")
        if not fve_path or not Path(fve_path).exists():
            return jsonify({"available": False})
        data = json.loads(Path(fve_path).read_text())
        return jsonify({"available": True, **data})
    except Exception:
        return jsonify({"available": False})


@app.route("/api/available_csvs")
def api_available_csvs():
    """List all CSVs available to load: default + any done run with an existing llmaaj_csv."""
    options = []
    # Default CSV
    if _DEFAULT_CSV and Path(_DEFAULT_CSV).exists():
        options.append({
            "label": Path(_DEFAULT_CSV).name,
            "path":  _DEFAULT_CSV,
            "run_id": None,
            "active": _csv_path == _DEFAULT_CSV,
        })
    # Done runs
    try:
        runs = rm.api_list_runs()
        for r in runs:
            if r.get("overall") != "done":
                continue
            art = rm.load_artifacts(r["run_id"])
            csv_p = art.get("llmaaj_csv", "")
            if not csv_p or not Path(csv_p).exists():
                continue
            if csv_p == _DEFAULT_CSV:
                continue  # already listed above
            options.append({
                "label":  Path(csv_p).name,
                "path":   csv_p,
                "run_id": r["run_id"],
                "active": _csv_path == csv_p,
            })
    except Exception:
        pass
    return jsonify(options)


@app.route("/api/runs/load_default_csv", methods=["POST"])
def api_load_default_csv():
    """Switch back to the default CSV."""
    if not _DEFAULT_CSV or not Path(_DEFAULT_CSV).exists():
        return jsonify({"error": "Default CSV not found"}), 404
    load_csv(_DEFAULT_CSV)
    return jsonify({"ok": True, "csv": Path(_DEFAULT_CSV).name, "n_features": len(_rows_full)})


@app.route("/api/runs/<run_id>/log")
def api_run_log(run_id):
    """Return the last N lines of pipeline.log for a run."""
    err, code = _validate_run_id(run_id)
    if err:
        return err, code
    rd = rm.run_dir(run_id)
    log_path = rd / "pipeline.log"
    try:
        n_lines = max(1, min(5000, int(request.args.get("lines", 500))))
    except (ValueError, TypeError):
        n_lines = 500
    if not log_path.exists():
        return jsonify({"lines": [], "exists": False})
    with open(log_path, "r", errors="replace") as f:
        all_lines = f.readlines()
    tail = all_lines[-n_lines:] if len(all_lines) > n_lines else all_lines
    return jsonify({"lines": [l.rstrip("\n") for l in tail], "exists": True,
                    "total_lines": len(all_lines)})


_STEP_LOG_PREFIX = {
    "00_generate_dataset": "00_gen",
    "01_collect_acts_A":   "01A_acts",
    "01_collect_acts_B":   "01B_acts",
    "02_train_crosscoder": "02_train",
    "03_collect_cc_acts":  "03_ccacts",
    "04_fve_dead_neurons": "04_fve",
    "04_llmaaj":           "04_llmaaj",
}

@app.route("/api/runs/<run_id>/step_log")
def api_step_log(run_id):
    """Return the SLURM .out log for a specific step (contains stdout+stderr via tee)."""
    err, code = _validate_run_id(run_id)
    if err:
        return err, code
    step = request.args.get("step", "")
    if step not in _STEP_LOG_PREFIX:
        return jsonify({"error": f"Unknown step: {step}"}), 400
    prefix = _STEP_LOG_PREFIX[step]
    logs_dir = rm.LOGS_DIR
    # Find the matching .out file: {run_id}_{prefix}_{job_id}.out
    pattern = f"{run_id}_{prefix}_*.out"
    matches = sorted(logs_dir.glob(pattern))
    if not matches:
        return jsonify({"lines": [], "exists": False, "file": None})
    log_path = matches[-1]  # latest if multiple (shouldn't happen)
    try:
        n_lines = max(1, min(5000, int(request.args.get("lines", 1000))))
    except (ValueError, TypeError):
        n_lines = 1000
    with open(log_path, "r", errors="replace") as f:
        all_lines = f.readlines()
    tail = all_lines[-n_lines:] if len(all_lines) > n_lines else all_lines
    return jsonify({"lines": [l.rstrip("\n") for l in tail], "exists": True,
                    "total_lines": len(all_lines), "file": log_path.name})


@app.route("/api/suggest_hyperparams")
def api_suggest_hyperparams():
    """Detect d_model + num_layers from a model ID and return suggested hyperparams."""
    model_id = request.args.get("model_a", "")
    if not model_id:
        return jsonify({"error": "missing model_a"}), 400
    d_model    = rm.detect_d_model(model_id)
    num_layers = rm.detect_num_layers(model_id)
    if d_model is None:
        return jsonify({"error": "could not detect d_model", "d_model": None})
    suggested = rm.suggest_hyperparams(d_model, topk=100)
    return jsonify({
        "d_model":    d_model,
        "num_layers": num_layers,
        "last_layer": (num_layers - 1) if num_layers else None,
        "n_features": suggested["n_features"],
        "aux_k":      suggested["aux_k"],
    })


@app.route("/api/model_layer_count")
def api_model_layer_count():
    """Return num_hidden_layers for a given model — used for layer validation."""
    model_id = request.args.get("model", "")
    if not model_id:
        return jsonify({"error": "missing model"}), 400
    n = rm.detect_num_layers(model_id)
    if n is None:
        return jsonify({"num_layers": None, "error": "could not detect"})
    return jsonify({"num_layers": n, "last_layer": n - 1})


@app.route("/api/runs/create", methods=["POST"])
def api_create_run():
    """Submit a new pipeline run from JSON body."""
    # Content-Type check
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    # Rate limit: max 3 creates per 60 seconds
    now = time.time()
    with _create_run_lock:
        _create_run_times[:] = [t for t in _create_run_times if now - t < 60]
        if len(_create_run_times) >= 3:
            return jsonify({"error": "rate limit exceeded — max 3 runs per 60 seconds"}), 429
        _create_run_times.append(now)

    body = request.get_json(silent=True) or {}
    required = ["name", "model_a", "model_b", "dataset_a", "dataset_b"]
    for k in required:
        if not body.get(k):
            return jsonify({"error": f"missing field: {k}"}), 400

    # Build a fake argparse namespace matching run_manager.cmd_create signature
    args = argparse.Namespace(
        name            = body.get("name", ""),
        model_a         = body["model_a"],
        model_b         = body["model_b"],
        dataset_a       = body["dataset_a"],
        dataset_b       = body["dataset_b"],
        layer_a         = int(body.get("layer_a", 25)),
        layer_b         = int(body["layer_b"]) if body.get("layer_b") is not None else None,
        expansion_factor = int(body.get("expansion_factor", 32)),
        topk            = int(body.get("topk", 100)),
        total_tokens    = int(body.get("total_tokens", 20_000_000)),
        judge_top_n     = int(body.get("judge_top_n", 999999)),
        d_model         = int(body["d_model"]) if body.get("d_model") else None,
        warmup          = int(body.get("warmup", 1000)),
        norm_init_scale = float(body.get("norm_init_scale", 1.0)),
        skip_dataset_gen = bool(body.get("skip_dataset_gen", False)),
        existing_dataset = body.get("existing_dataset", None),
    )
    if args.skip_dataset_gen and not args.existing_dataset:
        return jsonify({"error": "--skip-dataset-gen requires existing_dataset"}), 400

    try:
        run_id = rm.cmd_create(args)
        return jsonify({"run_id": run_id})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "internal error creating run"}), 500

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def page_viewer():
    return render_template('viewer.html')


@app.route("/runs")
def page_runs():
    return render_template('runs.html')


@app.route("/settings")
def page_settings():
    return render_template('settings.html')

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv",  default=_DEFAULT_CSV)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    a = p.parse_args()
    load_csv(a.csv)
    app.run(host=a.host, port=a.port, debug=False)
