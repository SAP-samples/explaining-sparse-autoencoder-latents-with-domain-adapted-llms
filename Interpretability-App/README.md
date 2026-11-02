# CrossCoder Interpretability Webapp

An end-to-end pipeline for training CrossCoders following Minder et al (https://arxiv.org/abs/2504.02922) on two language models and exploring the resulting features interactively.

Given two models (e.g. a base model and a domain-adapted version), the pipeline:
1. Generates a mixed activation dataset from two data sources 
2. Trains a CrossCoder
3. Classifies features via LLM-as-a-Judge + decoder norm analysis
4. Serves an interactive feature browser

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Interpretability-Webapp
pip install -r requirements.txt
```

> **CUDA note:** Install PyTorch with the right CUDA version for your system:
> ```bash
> # CUDA 12.4 (SAP cluster)
> pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> ```
> See [pytorch.org](https://pytorch.org) for other versions.

### 2. Set environment variables

Create a `.env` file in the project root:

```bash
AICORE_AUTH_URL=...       # SAP AI Core credentials (for LLM Judge step)
AICORE_CLIENT_ID=...
AICORE_CLIENT_SECRET=...
AICORE_BASE_URL=...
AICORE_RESOURCE_GROUP=...
```

### 3. Configure storage paths

By default, all run data (activations, checkpoints, CSVs, logs) is stored in `~/Webapp-Storage/`.

Override by setting environment variables:
```bash
export CC_STORAGE_ROOT=/path/to/your/storage   # run data (default: ~/Webapp-Storage)
export CC_VENV=/path/to/venv/bin/activate       # venv for SLURM jobs (default: ~/nova_venv/bin/activate)
```

---

## Running the Pipeline

### Option A — Web UI

Start the webapp and use the **Pipeline Runs** page to submit runs:

```bash
source <venv>/bin/activate
cd viewer/
python app.py --port 7861
```

Open `http://localhost:7861/runs` → **+ New Run** → fill in the form → Submit.

### Option B — CLI

```bash
python pipeline/run_manager.py create \
  --name        "my-run" \
  --model-a     "google/gemma-2-2b" \
  --model-b     "/path/to/finetuned-model" \
  --dataset-a   "HuggingFaceFW/fineweb" \
  --dataset-b   "HuggingFaceFW/fineweb-2" \
  --layer-a     13
```

All options:

| Flag | Required | Default | Description |
|---|---|---|---|
| `--name` | yes | — | Short name for this run |
| `--model-a` | yes | — | HF model ID or local path — base / reference model |
| `--model-b` | yes | — | HF model ID or local path — fine-tuned / CPT model |
| `--dataset-a` | yes | — | HF dataset ID or local path — distribution A |
| `--dataset-b` | yes | — | HF dataset ID or local path — distribution B |
| `--layer-a` | yes | — | Residual stream layer to hook |
| `--subset-a` | no | — | HF subset for dataset A (only for multi-config datasets) |
| `--subset-b` | no | — | HF subset for dataset B (only for multi-config datasets) |
| `--layer-b` | no | same as A | Layer for Model B |
| `--expansion-factor` | no | 32 | n\_features = factor × d\_model |
| `--topk` | no | 100 | Active features per token (k in BatchTopK) |
| `--total-tokens` | no | 20M | Dataset size (split 50/50 between A and B) |
| `--warmup` | no | 1000 | LR warmup steps |
| `--norm-init-scale` | no | 1.0 | Decoder norm init (paper default for BatchTopK) |
| `--skip-dataset-gen` | no | — | Skip Step 00, use `--existing-dataset` instead |
| `--existing-dataset` | no | — | Path to existing JSONL dataset (requires `--skip-dataset-gen`) |

Check progress:
```bash
# List all runs (shows run IDs)
python pipeline/run_manager.py list

# Detailed status for one run (run ID is printed by `create` and shown in `list`)
python pipeline/run_manager.py status --run-id run_<id>
```

---

## Pipeline Steps

```
Step 00  generate_dataset      mixed JSONL from two data sources (50/50 split)
Step 01A collect_acts_A        residual stream activations — Model A  ┐ parallel
Step 01B collect_acts_B        residual stream activations — Model B  ┘
Step 02  train_crosscoder      CrossCoder v5, BatchTopK sparsity
Step 03  collect_cc_acts       CrossCoder activations on test split
Step 04a llmaaj                LLM-as-a-Judge + Δnorm classification → CSV
Step 04b fve_dead_neurons      FVE + dead neuron metrics
```

All steps are submitted as SLURM jobs chained via `--dependency=afterok`.

---

## Starting the Feature Browser

```bash
source <venv>/bin/activate
cd viewer/
python app.py --port 7861
```

Then open `http://localhost:7861` in your browser.

> **On a remote cluster:** VS Code Remote SSH will auto-forward the port. If not, run:
> ```bash
> ssh -L 7861:<node-ip>:7861 <login-host>
> ```

### Feature Metrics

Each feature shows the following metrics in the **Decoder Norms & Classification** card:

| Metric | Description |
|---|---|
| `n_active` | Number of tokens (out of the test split) on which this feature fired — i.e. had a non-zero activation. A higher value means the feature is more commonly used. |
| `mean_act` | Mean activation value across all tokens where the feature fired (dead tokens excluded). Reflects the average strength of the feature when active. |
| `max_act` | Maximum activation value observed across the entire test split. Useful for spotting outlier tokens that trigger the feature very strongly. |
| `judge_score` | Overall LLM-as-a-Judge quality score (0–12). See [LLM-as-a-Judge](#llm-as-a-judge-labeling) below. |
| `lang_purity` | Sub-score (0–4): how consistently the feature activates on tokens. |
| `content_richness` | Sub-score (0–4): how semantically meaningful the activating tokens are (vs. punctuation, whitespace, etc.). |
| `specificity` | Sub-score (0–4): how specific/narrow the concept captured by the feature is (vs. generic high-frequency tokens). |

### Activation Quantiles

The **Activating Tokens by Quantile** grid divides the test-split tokens into 10 activation-strength buckets (0–10%, 10–20%, …, 90–100%) and shows representative tokens from each bucket. This lets you see:
- Which tokens fire the feature most strongly (top decile)
- Whether the feature has a clear semantic pattern or fires on diverse unrelated tokens
- How the concept shifts across activation strength levels

### LLM-as-a-Judge Labeling

The **label** and **explanation** shown for each feature are generated automatically by an LLM (Step 04a of the pipeline). The judge receives the tokens a feature actvates on and is asked to:
1. Assign a short descriptive label 
2. Write a one-sentence explanation of the concept
3. Score the feature on three dimensions: language purity, content richness, and specificity

The overall `judge_score` is a weighted average of these three sub-scores.

### Decoder Norm (Δnorm)

Δnorm measures which model's decoder relies more on a feature:

```
Δnorm = 0.5 × (1 + (‖W_dec_B‖ − ‖W_dec_A‖) / max(‖W_dec_A‖, ‖W_dec_B‖))
```

- **Δnorm ≈ 0** → feature is dominant in Model A (base model)
- **Δnorm ≈ 1** → feature is dominant in Model B (fine-tuned model)
- **Δnorm ≈ 0.5** → shared feature

The A/B/BOTH classification thresholds are adjustable in ⚙ Settings and applied live — no CSV regeneration needed.

---

## Directory Layout

```
Interpretability-Webapp/
├── pipeline/
│   ├── run_manager.py          # CLI + API for creating/tracking runs
│   └── templates/              # SLURM job templates
├── scripts/
│   ├── 00_generate_dataset/
│   ├── 01_collect_activations/
│   ├── 02_train_crosscoder/
│   ├── 03_collect_crosscoder_acts/
│   ├── 04_LLMaaJ/
│   └── 05_fve_dead_neurons/    # FVE + dead neuron computation
├── viewer/
│   ├── app.py                  # Flask webapp
│   ├── templates/              # HTML templates
│   └── static/                 # JS / CSS
├── requirements.txt
```
