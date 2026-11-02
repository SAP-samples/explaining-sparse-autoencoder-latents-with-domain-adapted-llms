#!/bin/bash
#SBATCH --job-name=sap-r50-base-expl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint='g548xlarge|g524xlarge'
#SBATCH --gres=gpu:a10g:2
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/sap_random50_base_expl_%j.log
#
# SAP | branching=3 | BASE explainer | CPT + BASE+RAG judges | random-50 features | sap-only | 20 tok/quantile

CPT_NODE="gpu-dy-g524xlarge-6"
CPT_NODE2="gpu-dy-g524xlarge-6"
BASE_NODE="gpu-dy-g524xlarge-5"
BASE_NODE2="gpu-dy-g524xlarge-5"
EXPL_NODE="gpu-dy-g524xlarge-5"
QDRANT_URL="http://cpu-dy-c59xlarge-1:6333"

META="/shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_activations/sae_acts_gemma3_CPTv6_sae_sae_base_cpt_sap_16x_l47_lr1e3_sae_dataset_modeldiff_sap_en10M_sap10M_20260708_layer47_test_20260818_201459_meta.pt"
SAE_CKPT="/shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_checkpoints/modeldiff_l47/sae_sae_base_cpt_sap_16x_l47_lr1e3.pt"
GENERATION_MODEL="/fsx/ns1/pretrained/huggingface/gemma3_CPTv6_IFTv2"

FEATURES="1890 2817 5866 8252 8940 13385 14902 15178 15650 15691 16250 16454 16939 16995 17914 19128 20165 20870 21038 21749 22108 23910 26091 26787 26804 26916 30334 30605 32414 33620 35593 39750 41739 42693 43356 44499 44863 44974 46766 47991 48061 48465 48477 49977 51072 52658 54902 58573 61235 61354"

source /shared/home/i777599/nova_venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python /shared/home/i777599/nova_model/SAE/scripts/paper_vllm_confidence_interval.py \
    --features $FEATURES \
    --meta "$META" \
    --sae-ckpt "$SAE_CKPT" \
    --generation-model "$GENERATION_MODEL" \
    --explainer-endpoint "http://${EXPL_NODE}:18003/v1" \
    --explainer-served-model "gemma-3-12b-it" \
    --rag-aware-prompt \
    --min-active 0 \
    --cpt-endpoints "http://${CPT_NODE}:18002/v1" "http://${CPT_NODE2}:18002/v1" \
    --cpt-served-model "cpt-v6-iftv2" \
    --qwen-endpoints "http://${BASE_NODE}:18003/v1" "http://${BASE_NODE2}:18003/v1" \
    --qwen-served-model "gemma-3-12b-it" \
    --judge-mode base_rag \
    --qdrant-url "$QDRANT_URL" \
    --qdrant-collection "sap_chunks" \
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
    --output "/shared/home/i777599/nova_model/SAE/results/tables/SAP/Random-50-features/sap_random50_base_expl.json"

echo "Finished: $(date)"
