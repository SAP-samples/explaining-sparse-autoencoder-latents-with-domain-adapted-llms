"""
run_manager.py  —  CrossCoder Pipeline Orchestrator

Creates and manages end-to-end pipeline runs. Each run:
  1. Generates a bilingual dataset (or skips if local path given)
  2. Collects residual-stream activations for Model A and Model B
  3. Trains a CrossCoder
  4. Collects CrossCoder activations (freq analysis)
  5. Runs LLM-as-a-Judge + norm/freq classification
  6. Computes FVE + dead-neuron metrics (eval split)

All SLURM jobs are chained via --dependency=afterok so the pipeline
is fully automated once submitted.

CLI:
    python run_manager.py create --model-a <id> --model-b <id> \
        --dataset-a <hf_or_path> --dataset-b <hf_or_path> --layer 25
    python run_manager.py status [--run-id <id>]
    python run_manager.py list
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── d_model detection ─────────────────────────────────────────────────────────

def _get_text_config(cfg):
    """
    For VLM wrappers (Gemma3/4 ForConditionalGeneration), the real language-model
    config is nested under cfg.text_config. Fall back to cfg itself for plain LMs.
    """
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and getattr(text_cfg, "hidden_size", None) is not None:
        return text_cfg
    return cfg


def detect_d_model(model_id: str) -> int | None:
    """
    Read hidden_size from a model's config.json without loading weights.
    Works for HF hub IDs (downloads config only) and local paths.
    Handles VLM wrappers (Gemma3/4) whose real config is under text_config.
    Returns None if detection fails.
    """
    try:
        from transformers import AutoConfig
        cfg = _get_text_config(AutoConfig.from_pretrained(model_id, trust_remote_code=False))
        for attr in ("hidden_size", "d_model", "n_embd"):
            v = getattr(cfg, attr, None)
            if v is not None:
                return int(v)
    except Exception as e:
        print(f"  [warn] d_model detection failed for {model_id}: {e}", file=sys.stderr)
    return None


def detect_num_layers(model_id: str) -> int | None:
    """Read num_hidden_layers from config.json without loading weights."""
    try:
        from transformers import AutoConfig
        cfg = _get_text_config(AutoConfig.from_pretrained(model_id, trust_remote_code=False))
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            v = getattr(cfg, attr, None)
            if v is not None:
                return int(v)
    except Exception as e:
        print(f"  [warn] num_layers detection failed for {model_id}: {e}", file=sys.stderr)
    return None


def estimate_model_params(model_id: str) -> int | None:
    """
    Estimate parameter count from AutoConfig without loading weights.
    Covers dense transformer architectures (Gemma, Llama, Mistral, Qwen, Falcon, …).
    Handles VLM wrappers (Gemma3/4) whose real LM config is under text_config.
    Returns total params as int, or None if detection fails.
    """
    try:
        from transformers import AutoConfig
        cfg = _get_text_config(AutoConfig.from_pretrained(model_id, trust_remote_code=False))

        H  = getattr(cfg, "hidden_size",         None) or getattr(cfg, "d_model", None) or getattr(cfg, "n_embd", None)
        L  = getattr(cfg, "num_hidden_layers",   None) or getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers", None)
        FF = getattr(cfg, "intermediate_size",   None)  # MLP dim (SwiGLU models have 2 linear gates)
        V  = getattr(cfg, "vocab_size",          None)
        KV_heads = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "n_head", None)
        Q_heads  = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)

        if H is None or L is None:
            return None

        # Attention: Q + K + V + O projections
        # For GQA: Q=H*H, K=V=H/n_groups*H, O=H*H
        kv_dim = H if KV_heads is None or Q_heads is None else H * KV_heads // Q_heads
        attn = H * H + 2 * H * kv_dim + H * H  # Q K V O

        # MLP: SwiGLU (gate + up + down) or standard (up + down)
        ff = FF or 4 * H
        is_swiglu = FF is not None and FF > 2 * H
        mlp = (3 * H * ff) if is_swiglu else (2 * H * ff)

        # Layer norms: ~2 per layer (pre-attn + pre-mlp), each 2*H params
        norms = 4 * H

        per_layer = attn + mlp + norms
        total = L * per_layer + (V * H if V else 0)  # + embeddings
        return int(total)
    except Exception as e:
        print(f"  [warn] param estimation failed for {model_id}: {e}", file=sys.stderr)
        return None


# GPU types available on this cluster, in ascending VRAM order.
# Use gres_type=None to request any GPU of that VRAM class (SLURM picks what's free).
_GPU_TIERS = [
    # (vram_gb, slurm_gres_type, partition)
    (24,  None,   "gpu"),   # any 24GB GPU — A10G or L4, whichever is free
]


def estimate_slurm_resources(model_id: str, step: str) -> dict:
    """
    Compute SLURM gres/mem requirements for a given model and pipeline step.

    Logic:
      1. Estimate model param count → BF16 weight size in GB
      2. Add overhead multiplier (KV cache, activations, gradients if training)
      3. Select minimum GPU tier that fits
      4. If model doesn't fit on one GPU, request multiple
      5. RAM = max(2× VRAM needed, 32G minimum)

    Returns dict with keys: gres_gpu (str), n_gpu (int), mem (str)
    """
    params = estimate_model_params(model_id)

    if params is None:
        # Safe fallback
        return {"gres_gpu": "gpu:a10g:1", "n_gpu": 1, "mem": "80G", "partition": "gpu"}

    params_b = params / 1e9
    # BF16: 2 bytes/param
    weight_gb = params_b * 2.0

    # Overhead multiplier per step:
    #   inference (activation collection): weights + activations ≈ 1.4×
    #   training (crosscoder): no model weights on GPU, just acts tensors → small
    if step in ("01_collect_acts_A", "01_collect_acts_B", "03_collect_cc_acts"):
        # Model loaded for inference: KV cache + activations ~ 40% overhead
        vram_needed = weight_gb * 1.4
    elif step == "02_train_crosscoder":
        # CrossCoder only — model not loaded, acts are pre-computed
        # n_features up to ~36k at d_model=2304 → ~200MB; very small
        vram_needed = 8.0  # fixed small budget for CC training
    else:
        # 04_llmaaj / 04_fve: loads model for tokenization/inference
        vram_needed = weight_gb * 1.4

    # Find minimum number of GPUs needed across available tiers
    n_gpu = 1
    chosen_tier = _GPU_TIERS[-1]  # default to largest
    for vram_gb, gres_type, partition in _GPU_TIERS:
        if vram_needed <= vram_gb:
            chosen_tier = (vram_gb, gres_type, partition)
            n_gpu = 1
            break
    else:
        # Doesn't fit on one GPU — use largest tier, compute how many
        largest_vram, largest_type, largest_partition = _GPU_TIERS[-1]
        n_gpu = int(math.ceil(vram_needed / largest_vram))
        n_gpu = min(n_gpu, 8)  # cap at 8 GPUs
        chosen_tier = (largest_vram, largest_type, largest_partition)

    # Add one extra GPU as headroom — peak VRAM during forward pass exceeds
    # the static weight estimate, causing OOM. Apply to all steps.
    n_gpu = min(n_gpu + 1, 8)

    gres_type  = chosen_tier[1]
    gres_vram  = chosen_tier[0]
    partition  = chosen_tier[2]

    # RAM: needs to hold the dataset + activations in CPU RAM
    # Rule: max(2× total VRAM allocated, 32G minimum, 16G per GPU)
    total_vram = gres_vram * n_gpu
    ram_gb = max(int(total_vram * 2), 32, 16 * n_gpu)
    # Round up to nearest 16G
    ram_gb = ((ram_gb + 15) // 16) * 16

    # Build gres string: gpu:<type>:<n> if type specified, else gpu:<n>
    if gres_type is None:
        gres_str = f"gpu:{n_gpu}"
    else:
        gres_str = f"gpu:{gres_type}:{n_gpu}"

    return {
        "gres_gpu":  gres_str,
        "n_gpu":     n_gpu,
        "mem":       f"{ram_gb}G",
        "partition": partition,
    }


def suggest_hyperparams(d_model: int, topk: int) -> dict:
    """
    Scaling-law heuristics based on Lindsey et al. 2025:
      n_features  = 16 × d_model   (expansion ratio)
      aux_k       = 2048            (paper value, fixed)
      LR          = 1e-3 × sqrt(total_tokens / 50M)  (computed at submit time)
    k (TopK) is always user-chosen — not auto-scaled.
    """
    n_features = 16 * d_model
    return {
        "n_features": n_features,
        "aux_k":      2048,
    }

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent.resolve()
STORAGE_ROOT = Path(os.environ.get("CC_STORAGE_ROOT", Path.home() / "Webapp-Storage"))

RUNS_DIR     = STORAGE_ROOT / "runs"       # run metadata + job scripts
DATA_DIR     = STORAGE_ROOT / "data"       # activations, CC acts, CSV, FVE
CKPT_DIR     = STORAGE_ROOT / "checkpoints"
LOGS_DIR     = STORAGE_ROOT / "logs"

SCRIPTS_DIR   = REPO_ROOT / "scripts"
PIPELINE_DIR  = Path(__file__).parent.resolve()
TEMPLATES_DIR = PIPELINE_DIR / "templates"

VENV     = os.environ.get("CC_VENV", str(Path.home() / "nova_venv" / "bin" / "activate"))
ENV_FILE = REPO_ROOT / ".env"

LA_TZ = timezone(timedelta(hours=-7))

# ── step names (order matters) ─────────────────────────────────────────────
STEPS = [
    "00_generate_dataset",
    "01_collect_acts_A",
    "01_collect_acts_B",
    "02_train_crosscoder",
    "03_collect_cc_acts",
    "04_llmaaj",
    "04_fve_dead_neurons",
]

# Steps that depend on the previous step(s)
# Maps step → list of steps it depends on
DEPENDENCIES = {
    "00_generate_dataset":   [],
    "01_collect_acts_A":     ["00_generate_dataset"],
    "01_collect_acts_B":     ["00_generate_dataset"],
    "02_train_crosscoder":   ["01_collect_acts_A", "01_collect_acts_B"],
    "03_collect_cc_acts":    ["02_train_crosscoder"],
    "04_llmaaj":             ["03_collect_cc_acts"],
    "04_fve_dead_neurons":   ["02_train_crosscoder"],
}


# ── helpers ────────────────────────────────────────────────────────────────────

def now_str() -> str:
    return datetime.now(LA_TZ).strftime("%Y%m%d_%H%M%S")


def make_run_id() -> str:
    ts = datetime.now(LA_TZ).strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"run_{ts}_{short}"


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def load_config(run_id: str) -> dict:
    p = run_dir(run_id) / "config.json"
    with open(p) as f:
        return json.load(f)


def load_status(run_id: str) -> dict:
    p = run_dir(run_id) / "status.json"
    defaults = {s: {"state": "pending", "job_id": None} for s in STEPS}
    if not p.exists():
        return defaults
    with open(p) as f:
        saved = json.load(f)
    # Merge saved state with defaults so newly-added steps appear correctly
    return {**defaults, **saved}


def save_status(run_id: str, status: dict):
    p = run_dir(run_id) / "status.json"
    with open(p, "w") as f:
        json.dump(status, f, indent=2)


def load_artifacts(run_id: str) -> dict:
    p = run_dir(run_id) / "artifacts.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def save_artifacts(run_id: str, artifacts: dict):
    p = run_dir(run_id) / "artifacts.json"
    with open(p, "w") as f:
        json.dump(artifacts, f, indent=2)


def sbatch(script_path: Path, dependency_job_ids: list[str] | None = None) -> str:
    """Submit a SLURM job. Returns the job ID string."""
    cmd = ["sbatch"]
    if dependency_job_ids:
        ids = ":".join(str(j) for j in dependency_job_ids if j)
        if ids:
            cmd += [f"--dependency=afterok:{ids}"]
    cmd.append(str(script_path))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # "Submitted batch job 12345"
    match = re.search(r"(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")
    return match.group(1)


def is_hf_id(s: str) -> bool:
    """True if s looks like a HuggingFace model/dataset ID (not a filesystem path)."""
    return "/" in s and not s.startswith("/") and not s.startswith(".")


def model_shortname(model_id: str) -> str:
    """gemma-2-2b or /fsx/.../gemma-2-2b → gemma-2-2b"""
    return Path(model_id).name.replace(" ", "_")


def render_template(template_name: str, ctx: dict) -> str:
    """Very simple {{var}} template rendering (no Jinja dependency)."""
    import re
    tmpl = (TEMPLATES_DIR / template_name).read_text()
    for k, v in ctx.items():
        tmpl = tmpl.replace("{{" + k + "}}", str(v))
    # Catch any unreplaced placeholders — fail loudly rather than submit broken scripts
    remaining = re.findall(r'\{\{[^}]+\}\}', tmpl)
    if remaining:
        raise ValueError(
            f"Template {template_name} has unreplaced vars: {remaining}\n"
            f"Context keys provided: {sorted(ctx.keys())}"
        )
    return tmpl


def write_job_script(run_id: str, step: str, content: str) -> Path:
    d = run_dir(run_id) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{step}.sh"
    p.write_text(content)
    p.chmod(0o755)
    return p


# ── create run ────────────────────────────────────────────────────────────────

def _validate_inputs(args):
    """Validate user inputs before creating any files or submitting jobs.

    Raises SystemExit with a clear message on any invalid input.
    """
    errors = []

    # Layer bounds check
    for model_id, layer, label in [
        (args.model_a, args.layer_a, "layer-a"),
        (args.model_b, args.layer_b, "layer-b"),
    ]:
        num_layers = detect_num_layers(model_id)
        if num_layers is not None:
            if layer < 0 or layer >= num_layers:
                errors.append(
                    f"  {label}={layer} is out of range for {model_id} "
                    f"(has {num_layers} layers, valid: 0–{num_layers - 1})"
                )

    # Existing dataset path check
    if args.skip_dataset_gen and args.existing_dataset:
        p = Path(args.existing_dataset)
        if not p.exists():
            errors.append(f"  --existing-dataset path not found: {p}")
        elif p.stat().st_size == 0:
            errors.append(f"  --existing-dataset file is empty: {p}")

    # Disk space pre-check (rough estimate: tokens × d_model × 4 bytes × 2 models + slack)
    try:
        import shutil
        free_gb = shutil.disk_usage(str(DATA_DIR)).free / 1e9
        # Per-model acts: tokens * d_model * 4 bytes; two models; plus CC acts (tokens*k*8) ~negligible
        d_model_hint = detect_d_model(args.model_a) or 2304
        tokens = getattr(args, "total_tokens", 20_000_000)
        estimated_gb = tokens * d_model_hint * 4 * 2 / 1e9 * 1.3  # 30% safety margin
        if estimated_gb > free_gb:
            errors.append(
                f"  Insufficient disk space on {DATA_DIR}: "
                f"need ~{estimated_gb:.1f} GB, have {free_gb:.1f} GB free"
            )
    except Exception:
        pass  # disk check is best-effort

    if errors:
        print("ERROR: Invalid arguments — pipeline NOT submitted:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)


def cmd_create(args):
    # Validate all inputs before touching disk
    _validate_inputs(args)

    run_id = make_run_id()
    rd = run_dir(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-detect d_model from Model A config (fast — downloads config.json only)
    d_model = args.d_model
    if d_model is None:
        print(f"  Detecting d_model from {args.model_a} ...", flush=True)
        d_model = detect_d_model(args.model_a)
        if d_model:
            print(f"  d_model = {d_model}", flush=True)
        else:
            print("  [warn] Could not detect d_model — defaulting to 2304 (Gemma-2-2b)", flush=True)
            d_model = 2304

    # Compute n_features from expansion_factor × d_model
    expansion_factor = getattr(args, "expansion_factor", None) or 16
    suggested = suggest_hyperparams(d_model, args.topk)
    n_features = expansion_factor * d_model
    aux_k      = suggested["aux_k"]

    print(f"  expansion_factor = {expansion_factor}  →  n_features = {n_features}  (factor × d_model={d_model})", flush=True)
    print(f"  aux_k      = {aux_k}  (paper value, fixed)", flush=True)
    print(f"  topk       = {args.topk}  (user-specified)", flush=True)

    data_dir = DATA_DIR / "runs" / run_id
    data_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id":       run_id,
        "name":         args.name,
        "created_at":   datetime.now(LA_TZ).isoformat(),
        "model_a":      args.model_a,
        "model_b":      args.model_b,
        "dataset_a":    args.dataset_a,
        "dataset_b":    args.dataset_b,
        "subset_a":     getattr(args, "subset_a", None),
        "subset_b":     getattr(args, "subset_b", None),
        "layer_a":      args.layer_a,
        "layer_b":      args.layer_b,
        "n_features":       n_features,
        "expansion_factor": expansion_factor,
        "topk":             args.topk,
        "aux_k":        aux_k,
        "total_tokens": args.total_tokens,
        "d_model":      d_model,
        "judge_top_n":  args.judge_top_n,
        "warmup":       getattr(args, "warmup", 1000),
        "norm_init_scale": getattr(args, "norm_init_scale", 1.0),
        "data_dir":     str(data_dir),
        "skip_dataset_gen": args.skip_dataset_gen,
        "existing_dataset": args.existing_dataset,
    }
    with open(rd / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Run ID : {run_id}")
    print(f"Dir    : {rd}")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print()

    # Build artifacts dict with expected output paths
    artifacts = _plan_artifacts(config)
    save_artifacts(run_id, artifacts)

    # Generate job scripts
    job_scripts = _generate_job_scripts(run_id, config, artifacts)

    # Submit SLURM chain
    status = {s: {"state": "pending", "job_id": None} for s in STEPS}
    job_ids: dict[str, str] = {}

    submit_order = [
        ("00_generate_dataset", []),
        ("01_collect_acts_A",   ["00_generate_dataset"]),
        ("01_collect_acts_B",   ["00_generate_dataset"]),
        ("02_train_crosscoder", ["01_collect_acts_A", "01_collect_acts_B"]),
        ("03_collect_cc_acts",  ["02_train_crosscoder"]),
        ("04_llmaaj",           ["03_collect_cc_acts"]),
        ("04_fve_dead_neurons", ["02_train_crosscoder"]),
    ]

    for step, dep_steps in submit_order:
        # Skip 00 if user provides existing dataset
        if step == "00_generate_dataset" and config["skip_dataset_gen"]:
            status[step]["state"] = "skipped"
            print(f"  [skip] {step}")
            continue

        dep_ids = [job_ids[s] for s in dep_steps if s in job_ids and job_ids[s]]
        # If a dep was skipped, don't include its (None) job_id
        dep_ids = [j for j in dep_ids if j is not None]

        try:
            jid = sbatch(job_scripts[step], dep_ids)
            job_ids[step] = jid
            status[step] = {"state": "queued", "job_id": jid}
            print(f"  [queued {jid}] {step}")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to submit {step}: {e.stderr}", file=sys.stderr)
            status[step] = {"state": "submit_failed", "job_id": None}

    save_status(run_id, status)
    print(f"\nRun submitted. Track with:  python pipeline/run_manager.py status --run-id {run_id}")
    return run_id


def _plan_artifacts(config: dict) -> dict:
    """Compute expected output file paths for all pipeline steps."""
    run_id  = config["run_id"]
    data_dir = Path(config["data_dir"])
    ckpt_dir = CKPT_DIR

    ma_short = model_shortname(config["model_a"])
    mb_short = model_shortname(config["model_b"])
    layer_a  = config["layer_a"]
    layer_b  = config["layer_b"]
    ts       = run_id.replace("run_", "")

    dataset_path = (
        str(data_dir / f"dataset_{ts}.jsonl")
        if not config["skip_dataset_gen"]
        else config["existing_dataset"]
    )

    acts_a   = str(data_dir / f"acts_{ma_short}_layer{layer_a}_{ts}.pt")
    acts_b   = str(data_dir / f"acts_{mb_short}_layer{layer_b}_{ts}.pt")
    cc_ckpt  = str(ckpt_dir / f"cc_{run_id}_f{config['n_features']}_k{config['topk']}.pt")
    cc_acts_meta = str(data_dir / f"cc_acts_{ts}_meta.pt")
    llmaaj_csv   = str(data_dir / f"features_{ts}.csv")
    fve_json     = str(data_dir / f"fve_{ts}.json")
    run_log      = str(run_dir(run_id) / "pipeline.log")

    return {
        "dataset":        dataset_path,
        "acts_a":         acts_a,
        "acts_b":         acts_b,
        "cc_checkpoint":  cc_ckpt,
        "cc_acts_meta":   cc_acts_meta,
        "llmaaj_csv":     llmaaj_csv,
        "fve_json":       fve_json,
        "run_log":        run_log,
    }


def _generate_job_scripts(run_id: str, config: dict, artifacts: dict) -> dict[str, Path]:
    """Render all SLURM job scripts for this run and return step → Path map."""
    log_base = str(LOGS_DIR)

    # Pre-compute resources for steps that load the full model vs. CC-only
    # Use the larger of the two models as the bottleneck for inference steps
    print("  Estimating SLURM resources from model configs...", flush=True)
    # For inference steps (acts collection, cc acts, judge): use larger model
    # We estimate both and pick the one that needs more VRAM
    res_a = estimate_slurm_resources(config["model_a"], "01_collect_acts_A")
    res_b = estimate_slurm_resources(config["model_b"], "01_collect_acts_B")
    # Pick whichever needs more GPUs; if equal, pick a (arbitrary tiebreak)
    res_inference = res_a if res_a["n_gpu"] >= res_b["n_gpu"] else res_b

    res_train = estimate_slurm_resources(config["model_a"], "02_train_crosscoder")

    def _gres_line(res: dict) -> str:
        return res["gres_gpu"]  # already "gpu:a10g:1" format

    print(f"    inference steps : gres={_gres_line(res_inference)}  mem={res_inference['mem']}  partition={res_inference['partition']}", flush=True)
    print(f"    training step   : gres={_gres_line(res_train)}  mem={res_train['mem']}  partition={res_train['partition']}", flush=True)

    ctx_base = {
        "run_id":         run_id,
        "venv":           VENV,
        "env_file":       str(ENV_FILE),
        "log_dir":        log_base,
        "scripts_dir":    str(SCRIPTS_DIR),
        "pipeline_dir":   str(PIPELINE_DIR),
        "model_a":        config["model_a"],
        "model_b":        config["model_b"],
        "model_a_short":  model_shortname(config["model_a"]),
        "model_b_short":  model_shortname(config["model_b"]),
        "layer_a":        config["layer_a"],
        "layer_b":        config["layer_b"],
        "n_features":     config["n_features"],
        "topk":           config["topk"],
        "aux_k":          config["aux_k"],
        "total_tokens":   config["total_tokens"],
        "warmup":         config.get("warmup", 1000),
        "norm_init_scale": config.get("norm_init_scale", 1.0),
        "judge_top_n":    config.get("judge_top_n", 999999),
        "dataset_a":      config.get("dataset_a", ""),
        "dataset_b":      config.get("dataset_b", ""),
        "subset_a_arg":   f'--subset-a   "{config["subset_a"]}" \\\n    ' if config.get("subset_a") else "",
        "subset_b_arg":   f'--subset-b   "{config["subset_b"]}" \\\n    ' if config.get("subset_b") else "",
        # Resource placeholders — overridden per-step below
        "gres_gpu":       _gres_line(res_inference),
        "mem":            res_inference["mem"],
        "partition":      res_inference["partition"],
        "cpus_per_task":  8 * res_inference["n_gpu"],
        **{k: v for k, v in artifacts.items()},
    }

    # Per-step resource overrides (cpus_per_task scales with n_gpu: 8 per GPU)
    def _cpus(n_gpu: int) -> int:
        return 8 * n_gpu

    step_resources = {
        "00_generate_dataset": {"gres_gpu": "gpu:a10g:1",             "mem": "32G",                 "partition": "gpu", "cpus_per_task": 4},
        "01_collect_acts_A":   {"gres_gpu": _gres_line(res_inference), "mem": res_inference["mem"], "partition": res_inference["partition"], "cpus_per_task": _cpus(res_inference["n_gpu"])},
        "01_collect_acts_B":   {"gres_gpu": _gres_line(res_inference), "mem": res_inference["mem"], "partition": res_inference["partition"], "cpus_per_task": _cpus(res_inference["n_gpu"])},
        "02_train_crosscoder": {"gres_gpu": _gres_line(res_train),     "mem": res_train["mem"],     "partition": res_train["partition"],     "cpus_per_task": _cpus(res_train["n_gpu"])},
        "03_collect_cc_acts":  {"gres_gpu": _gres_line(res_inference), "mem": res_inference["mem"], "partition": res_inference["partition"], "cpus_per_task": _cpus(res_inference["n_gpu"])},
        "04_llmaaj":           {"gres_gpu": _gres_line(res_inference), "mem": res_inference["mem"], "partition": res_inference["partition"], "cpus_per_task": _cpus(res_inference["n_gpu"])},
        "04_fve_dead_neurons": {"gres_gpu": _gres_line(res_inference), "mem": res_inference["mem"], "partition": res_inference["partition"], "cpus_per_task": _cpus(res_inference["n_gpu"])},
    }

    scripts = {}
    for step in STEPS:
        if step == "00_generate_dataset" and config["skip_dataset_gen"]:
            continue
        tmpl_name = f"{step}.sh.tmpl"
        ctx = {**ctx_base, **step_resources.get(step, {})}
        content = render_template(tmpl_name, ctx)
        scripts[step] = write_job_script(run_id, step, content)

    return scripts


def _compute_overall_state(status: dict) -> str:
    """Compute overall pipeline state from individual step states."""
    states = [v["state"] for v in status.values()]
    TERMINAL_FAIL = {"failed", "cancelled", "submit_failed"}
    if any(s in TERMINAL_FAIL for s in states):
        return "failed"
    if all(s in ("done", "skipped") for s in states):
        return "done"
    if any(s == "running" for s in states):
        return "running"
    if any(s == "queued" for s in states):
        return "queued"
    return "pending"


# ── status ────────────────────────────────────────────────────────────────────

_squeue_cache: dict = {"ts": 0.0, "live": {}}
_SQUEUE_TTL = 10.0  # seconds


def _query_squeue(job_ids: list[str]) -> dict:
    """Query squeue for the given job IDs, with a 10-second shared cache."""
    now = time.time()
    if now - _squeue_cache["ts"] < _SQUEUE_TTL:
        return _squeue_cache["live"]
    try:
        result = subprocess.run(
            ["squeue", "--jobs", ",".join(job_ids), "--format=%i %T", "--noheader"],
            capture_output=True, text=True, timeout=10,
        )
        live = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) == 2:
                live[parts[0]] = parts[1]
        _squeue_cache["ts"] = now
        _squeue_cache["live"] = live
        return live
    except Exception:
        return _squeue_cache["live"]  # stale cache better than nothing


def _refresh_slurm_states(status: dict) -> dict:
    """Query squeue for all queued/running jobs and update state."""
    job_ids = [v["job_id"] for v in status.values() if v.get("job_id")]
    if not job_ids:
        return status

    live = _query_squeue(job_ids)

    SLURM_TO_STATE = {
        "PENDING":   "queued",
        "CONFIGURING": "queued",
        "RUNNING":   "running",
        "COMPLETING": "running",
        "COMPLETED": "done",
        "FAILED":    "failed",
        "CANCELLED": "cancelled",
        "TIMEOUT":   "failed",
        "NODE_FAIL": "failed",
    }

    for step, info in status.items():
        jid = info.get("job_id")
        if jid and jid in live:
            slurm_state = live[jid]
            new_state = SLURM_TO_STATE.get(slurm_state, info["state"])
            status[step]["state"] = new_state
        elif jid and jid not in live and info["state"] in ("queued", "running"):
            # Not in squeue anymore → completed or failed; check sacct
            status[step]["state"] = _sacct_state(jid)

    return status


def _sacct_state(job_id: str) -> str:
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True, text=True, timeout=10
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if lines:
            s = lines[0].split("|")[0].split()[0].upper()
            if "COMPLETED" in s:
                return "done"
            if "FAILED" in s or "CANCEL" in s or "TIMEOUT" in s:
                return "failed"
    except Exception:
        pass
    return "unknown"


def cmd_status(args):
    if args.run_id:
        run_ids = [args.run_id]
    else:
        run_ids = sorted([p.name for p in RUNS_DIR.iterdir() if p.is_dir()])

    for run_id in run_ids:
        try:
            config = load_config(run_id)
        except FileNotFoundError:
            print(f"{run_id}: no config found")
            continue

        status = load_status(run_id)
        status = _refresh_slurm_states(status)
        save_status(run_id, status)

        STATE_ICON = {
            "pending":      "○",
            "queued":       "⏳",
            "running":      "▶",
            "done":         "✓",
            "failed":       "✗",
            "skipped":      "–",
            "cancelled":    "✗",
            "unknown":      "?",
            "submit_failed": "✗",
        }

        print(f"\n{'─'*60}")
        print(f"Run  : {run_id}")
        if config.get("name"):
            print(f"Name : {config['name']}")
        print(f"A    : {config['model_a']}")
        print(f"B    : {config['model_b']}")
        print(f"LayerA: {config.get('layer_a', config.get('layer', '?'))}  LayerB: {config.get('layer_b', config.get('layer', '?'))}  features: {config['n_features']}  k: {config['topk']}")
        print()
        for step in STEPS:
            info = status.get(step, {"state": "pending", "job_id": None})
            icon = STATE_ICON.get(info["state"], "?")
            jid  = f"  [job {info['job_id']}]" if info.get("job_id") else ""
            print(f"  {icon}  {step}{jid}  ({info['state']})")


def cmd_list(_args):
    if not RUNS_DIR.exists():
        print("No runs yet.")
        return
    runs = sorted([p.name for p in RUNS_DIR.iterdir() if p.is_dir()])
    if not runs:
        print("No runs yet.")
        return
    for run_id in runs:
        try:
            config = load_config(run_id)
            status = load_status(run_id)
            overall = _compute_overall_state(status)
            la = model_shortname(config["model_a"])
            lb = model_shortname(config["model_b"])
            layer_a = config.get("layer_a", config.get("layer", "?"))
            layer_b = config.get("layer_b", config.get("layer", "?"))
            layer_str = f"layer={layer_a}" if layer_a == layer_b else f"layerA={layer_a} layerB={layer_b}"
            name = config.get("name", "")
            name_str = f'  "{name}"' if name else ""
            print(f"{run_id}  [{overall:8}]  {la} vs {lb}  {layer_str}{name_str}")
        except Exception as e:
            print(f"{run_id}  [error: {e}]")


def _enrich_artifacts(artifacts: dict) -> dict:
    """Add exists flag to each artifact entry."""
    return {
        k: {"path": v, "exists": bool(v and Path(v).exists())}
        for k, v in artifacts.items()
    }


# ── API helpers (called from Flask) ─────────────────────────────────────────

def api_get_run(run_id: str) -> dict:
    """Return full run info dict for the viewer."""
    config    = load_config(run_id)
    status    = load_status(run_id)
    status    = _refresh_slurm_states(status)
    save_status(run_id, status)
    artifacts = _enrich_artifacts(load_artifacts(run_id))

    overall = _compute_overall_state(status)

    return {
        "run_id":    run_id,
        "config":    config,
        "status":    status,
        "artifacts": artifacts,
        "overall":   overall,
    }


def api_list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for p in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        try:
            config    = load_config(p.name)
            status    = load_status(p.name)
            artifacts = _enrich_artifacts(load_artifacts(p.name))

            overall = _compute_overall_state(status)

            runs.append({
                "run_id":    p.name,
                "config":    config,
                "status":    status,
                "artifacts": artifacts,
                "overall":   overall,
            })
        except Exception:
            pass
    return runs


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CrossCoder Pipeline Manager")
    sub = parser.add_subparsers(dest="cmd")

    # create
    p_create = sub.add_parser("create", help="Create and submit a new pipeline run")
    p_create.add_argument("--name",         required=True, help="Human-readable run name (e.g. 'gemma2-swallow-layer25')")
    p_create.add_argument("--model-a",      required=True, help="HF ID or cluster path for Model A (base)")
    p_create.add_argument("--model-b",      required=True, help="HF ID or cluster path for Model B (CPT)")
    p_create.add_argument("--dataset-a",    required=True, help="HF dataset ID or path — language A (e.g. English)")
    p_create.add_argument("--subset-a",     default=None,  help="HF subset for dataset-a (e.g. 'sample-10BT')")
    p_create.add_argument("--dataset-b",    required=True, help="HF dataset ID or path — language B (e.g. Japanese)")
    p_create.add_argument("--subset-b",     default=None,  help="HF subset for dataset-b (e.g. 'jpn_Jpan')")
    p_create.add_argument("--layer-a",      type=int, default=25, help="Residual stream layer for Model A")
    p_create.add_argument("--layer-b",      type=int, default=None, help="Residual stream layer for Model B (default: same as --layer-a)")
    p_create.add_argument("--expansion-factor", type=int, default=16, help="n_features = expansion_factor × d_model (paper default: 16)")
    p_create.add_argument("--judge-top-n",     type=int, default=999999, help="Max features to judge via LLM (default: all). Use 100 for quick tests.")
    p_create.add_argument("--topk",         type=int, default=100, help="BatchTopK k (active features per token)")
    p_create.add_argument("--total-tokens", type=int, default=20_000_000, help="Total dataset tokens (split 50/50)")
    p_create.add_argument("--d-model",      type=int, default=None, help="Hidden dim (auto-detected if None)")
    p_create.add_argument("--warmup",       type=int, default=1000, help="LR warmup steps (paper default: 1000)")
    p_create.add_argument("--norm-init-scale", type=float, default=1.0, help="Decoder norm init scale (BatchTopK paper default: 1.0)")
    p_create.add_argument("--skip-dataset-gen", action="store_true",
                          help="Skip dataset generation (use --existing-dataset)")
    p_create.add_argument("--existing-dataset", default=None,
                          help="Path to existing JSONL dataset (requires --skip-dataset-gen)")

    # status
    p_status = sub.add_parser("status", help="Show pipeline run status")
    p_status.add_argument("--run-id", default=None, help="Show specific run (default: all)")

    # list
    sub.add_parser("list", help="List all runs")

    args = parser.parse_args()
    if args.cmd == "create":
        if args.skip_dataset_gen and not args.existing_dataset:
            parser.error("--skip-dataset-gen requires --existing-dataset")
        if args.layer_b is None:
            args.layer_b = args.layer_a
        cmd_create(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
