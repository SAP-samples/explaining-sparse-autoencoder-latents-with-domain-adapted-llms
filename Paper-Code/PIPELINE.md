# SAE Evaluation Pipeline

End-to-end pipeline for the SAE-based domain knowledge evaluation.
Two parallel tracks: **SAP** (Gemma-3-12B) and **Fintrain/Salesforce** (Llama-3-8B).

---

## Overview

```
00_generate_dataset      → create token-balanced input JSONL
01_vllm_endpoints        → start LLM inference servers (needed for step 07)
02_rag_setup             → build Qdrant hybrid-search index (needed for step 07)
03_collect_activations   → collect residual stream activations from BASE + CPT models
04_train_sae             → train JumpReLU SAE on those activations
05_collect_sae_activations → run test set through trained SAE → per-feature activation files
06_llm_judge_pipeline    → single-feature pilot: generate explanations + LLM-judge scores
07_big_run               → full bootstrap evaluation over 100 features (all conditions)
08_compute_scores        → compute bootstrap confidence intervals from JSON results
```

---

## Step 00 — Generate Dataset

**Purpose:** Create the token-balanced input JSONL used throughout the pipeline.
Each file contains EN tokens and domain-specific tokens (SAP or Salesforce) interleaved.

| Script | What it does |
|---|---|
| `generate_dataset_modeldiff_sap.sh` | Generates SAP dataset (`en10M_sap10M`) |
| `generate_dataset_modeldiff_salesforce.sh` | Generates Salesforce dataset (`en10M_salesforce10M`) |

---

## Step 01 — vLLM Endpoints

**Purpose:** Start persistent vLLM Docker containers serving the models needed for LLM-as-Judge in step 07.
Run these on dedicated CPU/GPU nodes and keep them alive for the duration of the big run.

| Script | Model | Port |
|---|---|---|
| `vllm_base.sh` | `google/gemma-3-12b-it` (SAP BASE) | 18003 |
| `vllm_base_2.sh` | `google/gemma-3-12b-it` (SAP BASE, replica) | 18008 |
| `vllm_cpt.sh` | `gemma3_CPTv6_IFTv2` (SAP CPT) | 18002 |
| `vllm_cpt_2.sh` | `gemma3_CPTv6_IFTv2` (SAP CPT, replica) | 18009 |
| `vllm_base_fintrain_1.sh` | `Meta-Llama-3-8B-Instruct` (Fin BASE) | 18004 |
| `vllm_base_fintrain_2.sh` | `Meta-Llama-3-8B-Instruct` (Fin BASE, replica) | 18005 |
| `vllm_cpt_fintrain_1.sh` | `Llama-Fin-8b` (Fin CPT) | 18006 |
| `vllm_cpt_fintrain_2.sh` | `Llama-Fin-8b` (Fin CPT, replica) | 18007 |

`vllm_router.py` — optional load balancer across replicas.

---

## Step 02 — RAG Setup

**Purpose:** Build the Qdrant hybrid-search index (dense + sparse, BGE-M3) used as RAG baseline in step 07.
One-time setup — servers stay running persistently.
See `02_rag_setup/README.md` for full details.

**SAP:** `create_chunks_sap.sh` → `embed_shards_sap.sh` → `merge_vector_db_sap.sh` → `qdrant_server_sap.sh` → `build_qdrant_fast_sap.sh`

**Fintrain:** `create_chunks_fintrain.sh` → `embed_shards_fintrain.sh` → `merge_vector_db_fintrain.sh` → `qdrant_server_fintrain.sh` → `build_qdrant_fast_fintrain.sh`

---

## Step 03 — Collect Activations

**Purpose:** Run the BASE and CPT models over the input dataset and save residual stream
activations at the target layer as `.pt` memory-mapped files.

Must be run **before** step 04 (SAE training needs these as input).

### SAP (Gemma-3-12B, layer 47)

| Script | Model | Lang | Output |
|---|---|---|---|
| `collect_acts_base-model_train-set_en-lang_SAP.sh` | `gemma-3-12b-pt` | EN | `acts_gemma-3-12b-pt_..._en_....pt` |
| `collect_acts_cpt-model_train-set_sap-lang_SAP.sh` | `gemma3_CPTv6` | SAP | `acts_gemma3_CPTv6_..._sap_....pt` |

### Fintrain (Llama-3-8B, layer 31)

| Script | Model | Lang | Output |
|---|---|---|---|
| `collect_acts_base-model_train-set_en-lang_SALESFORCE.sh` | `Meta-Llama-3-8B-Instruct` | EN | `acts_Meta-Llama-3-8B-Instruct_..._en_....pt` |
| `collect_acts_cpt-model_train-set_salesforce-lang_SALESFORCE.sh` | `Llama-Fin-8b` | Salesforce | `acts_..._salesforce_....pt` |

---

## Step 04 — Train SAE

**Purpose:** Train a JumpReLU Sparse Autoencoder on the collected activations.
BASE SAE is trained first; CPT SAE warm-starts from the BASE SAE checkpoint.

### SAP

```
1. sbatch train_sae_gemma3_BASE-MODEL_ENG-DATA.sh
       input:  acts_gemma-3-12b-pt_..._en_....pt
       output: SAP_modeldiff/sae_checkpoints/modeldiff_l47/sae_sae_base_16x_l47_lr1e3.pt

2. sbatch train_sae_gemma3_CPT-MODEL_SAP-DATA.sh
       input:  acts_gemma3_CPTv6_..._sap_....pt
       init:   sae_sae_base_16x_l47_lr1e3.pt  (warm-start)
       output: SAP_modeldiff/sae_checkpoints/modeldiff_l47/sae_sae_base_cpt_sap_16x_l47_lr1e3.pt
```

### Fintrain

```
1. sbatch train_sae_llama3_BASE-MODEL_ENG-DATA.sh
       input:  acts_Meta-Llama-3-8B-Instruct_..._en_....pt
       output: Salesforce_modeldiff/sae_checkpoints/sae_sae_base_salesforce_20M_16x_l31.pt

2. sbatch train_sae_llama3_CPT-MODEL_SALESFORCE-DATA.sh
       input:  acts_..._salesforce_....pt
       init:   sae_sae_base_salesforce_20M_16x_l31.pt  (warm-start)
       output: Salesforce_modeldiff/sae_checkpoints/sae_sae_cpt_salesforce_20M_16x_l31.pt
```

**Hyperparameters:**

| | SAP | Fintrain |
|---|---|---|
| Base model | Gemma-3-12B | Meta-Llama-3-8B-Instruct |
| Layer (0-indexed) | 47 | 31 |
| n_features | 61,440 | 65,536 |
| lr | 1e-3 | 2e-4 |
| λ sparsity | 1e-4 | 2e-4 |
| L0 target | 150 | 150 |
| Batch | 4096 | 2048 |
| Epochs | 10 | 10 |

---

## Step 05 — Collect SAE Activations

**Purpose:** Run the **test split** of the dataset through the trained CPT SAE to get
per-feature activation statistics. Output is a `_meta.pt` file used by step 07.

| Script | Domain | Model | SAE |
|---|---|---|---|
| `collect_sae_activations_SAE-CPT_MODEL-CPT_sap.sh` | SAP | `gemma3_CPTv6` | `sae_sae_base_cpt_sap_16x_l47_lr1e3.pt` |
| `collect_sae_activations_SAE-CPT_MODEL-CPT_salesforce.sh` | Fintrain | `Llama-Fin-8b` | `sae_sae_cpt_salesforce_20M_16x_l31.pt` |

---

## Step 06 — LLM Judge Pipeline (Single-Feature Pilot)

**Purpose:** Pilot step — runs the full explanation + LLM-judge evaluation for a
**single feature at a time**. Useful for debugging or spot-checking before the big run.

| Script | Domain |
|---|---|
| `singlesae_analysis_sap.sh` | SAP |
| `singlesae_salesforce_analysis.sh` | Fintrain |

Requires vLLM endpoints (step 01) to be running.

---

## Step 07 — Big Run (Bootstrap Evaluation)

**Purpose:** Full evaluation over **100 selected features**, all conditions, with bootstrap
resampling for statistical significance. Each script is one SLURM job — submit all
conditions in parallel.

Requires: step 01 (vLLM endpoints) + step 02 (Qdrant) + step 05 (SAE activations meta file).

### SAP conditions (use `paper_vllm_confidence_interval.py`)

| Script | Condition |
|---|---|
| `bootstrap_br3_base_expl_cptmeta_saponly_tpq20.sh` | BASE explainer |
| `bootstrap_br3_base_expl_rag_cptmeta_saponly_tpq20.sh` | BASE explainer + RAG |
| `bootstrap_br3_cpt_expl_cptmeta_saponly_tpq20.sh` | CPT explainer |
| `bootstrap_br3_cpt_expl_rag_cptmeta_saponly_tpq20.sh` | CPT explainer + RAG |
| `bootstrap_br3_sota_expl_cptmeta_saponly_tpq20.sh` | SOTA explainer |
| `bootstrap_br3_sota_expl_rag_cptmeta_saponly_tpq20.sh` | SOTA explainer + RAG |
| `bootstrap_br3_dummy_cptmeta.sh` | DUMMY baseline |
| `sap_random50_*.sh` | Same conditions, random 50-feature subset |

### Fintrain conditions (use `paper_vllm_confidence_interval_fintrain.py`)

| Script | Condition |
|---|---|
| `bootstrap_ft_br3_base_expl_salesforceonly.sh` | BASE explainer |
| `bootstrap_ft_br3_base_expl_rag_salesforceonly.sh` | BASE explainer + RAG |
| `bootstrap_ft_br3_cpt_expl_salesforceonly.sh` | CPT explainer |
| `bootstrap_ft_br3_cpt_expl_rag_salesforceonly.sh` | CPT explainer + RAG |
| `bootstrap_ft_br3_sota_expl_salesforceonly.sh` | SOTA explainer |
| `bootstrap_ft_br3_sota_expl_rag_salesforceonly.sh` | SOTA explainer + RAG |
| `bootstrap_ft_br3_dummy_salesforceonly.sh` | DUMMY baseline |
| `ft_random50_*.sh` | Same conditions, random 50-feature subset |

Results are written to `/shared/home/i777599/nova_model/SAE/results/`.

---

## Step 08 — Compute Scores

**Purpose:** Aggregate the per-feature JSON results from step 07 and compute
bootstrap confidence intervals.

```bash
python compute_bootstrap_ci.py \
    --input-dir  /path/to/results/json/files \
    --output-dir /path/to/output \
    --bootstrap combined
```

Works for both SAP and Fintrain — pass the correct `--input-dir`.

---

## Dependency Graph

```
00 ──→ 03 ──→ 04 ──→ 05 ──→ 07 ──→ 08
                              ↑
01 ───────────────────────────┤
02 ───────────────────────────┘
06 (optional pilot, same deps as 07)
```
