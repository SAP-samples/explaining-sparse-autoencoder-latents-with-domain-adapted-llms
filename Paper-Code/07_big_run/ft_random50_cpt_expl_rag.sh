#!/bin/bash
#SBATCH --job-name=ft-r50-cpt-expl-rag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint='g548xlarge|g524xlarge'
#SBATCH --gres=gpu:a10g:2
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/ft_random50_cpt_expl_rag_%j.log
#
# Fintech | branching=3 | CPT+RAG explainer | CPT + BASE+RAG judges | random-50 features | domain-only tokens

CPT_NODE1="gpu-dy-g524xlarge-4"
BASE_NODE1="gpu-dy-g524xlarge-3"
EXPL_NODE="gpu-dy-g524xlarge-4"
QDRANT_URL="http://cpu-dy-c59xlarge-2:6333"

META="/shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_activations/sae_acts_Llama-Fin-8b_sae_sae_cpt_salesforce_20M_16x_l31_sae_dataset_modeldiff_salesforce_en10M_salesforce10M_20260828_layer31_test_20260831_094209_meta.pt"
SAE_CKPT="/shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_checkpoints/sae_sae_cpt_salesforce_20M_16x_l31.pt"
GENERATION_MODEL="/fsx/i777599/models/Llama-Fin-8b"

FEATURES="183 186 408 1221 1481 3632 3976 5907 7352 7426 15010 15507 19396 21165 21635 21657 26123 26220 27324 30026 30802 34536 35266 35595 36828 36856 37098 37405 39197 39684 39787 40147 40881 41295 41919 43298 44632 47997 48876 49540 52939 55945 58756 59066 60023 60599 63153 63958 64454 64706"

echo "============================================================"
echo "  Fintrain br3 | CPT+RAG explainer | CPT + BASE+RAG judges | random-50 features | domain-only tokens"
echo "  Node: $(hostname)  Started: $(date)"
echo "============================================================"

source /shared/home/i777599/nova_venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python /shared/home/i777599/nova_model/SAE/scripts/paper_vllm_confidence_interval_fintrain.py \
    --features $FEATURES \
    --meta "$META" \
    --sae-ckpt "$SAE_CKPT" \
    --generation-model "$GENERATION_MODEL" \
    --explainer-endpoint "http://${EXPL_NODE}:18006/v1" \
    --explainer-served-model "llama-fin-8b" \
    --explainer-rag \
    --rag-aware-prompt \
    --min-active 0 \
    --cpt-endpoints "http://${CPT_NODE1}:18006/v1" \
    --cpt-served-model "llama-fin-8b" \
    --qwen-endpoints "http://${BASE_NODE1}:18004/v1" \
    --qwen-served-model "llama-3-8b-instruct" \
    --judge-mode base_rag \
    --qdrant-url "$QDRANT_URL" \
    --qdrant-collection "fintrain_chunks" \
    --rag-top-k 5 \
    --n-quantile-samples 3 \
    --n-desc-samples 3 \
    --n-metric-samples 3 \
    --temperature 0.7 \
    --gen-tokens 20 \
    --generation-max-tokens 256 \
    --explainer-max-tokens 256 \
    --explainer-tokens-per-quantile 20 \
    --held-out-per-quantile 5 \
    --per-instance 32 \
    --output "/shared/home/i777599/nova_model/SAE/results/tables/Salesforce/Random-50-features/ft_random50_cpt_expl_rag.json"

echo "Finished: $(date)"
