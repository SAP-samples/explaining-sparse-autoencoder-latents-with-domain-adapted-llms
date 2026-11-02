#!/bin/bash
#SBATCH --job-name=br3-base-cptmeta-tpq20
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint='g524xlarge'
#SBATCH --gres=gpu:a10g:2
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/br3_base_cptmeta_saponly_tpq20_%j.log
#
# branching=3 | BASE explainer | CPT + BASE+RAG judges | 100 features | CPT meta | sap-only | 20 tok/quantile

CPT_NODE="gpu-dy-g524xlarge-20"
CPT_NODE2="gpu-dy-g524xlarge-14"
BASE_NODE="gpu-dy-g524xlarge-16"
BASE_NODE2="gpu-dy-g524xlarge-13"
EXPL_NODE="gpu-dy-g524xlarge-16"
QDRANT_URL="http://cpu-dy-c59xlarge-1:6333"

META="/shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_activations/sae_acts_gemma3_CPTv6_sae_sae_base_cpt_sap_16x_l47_lr1e3_sae_dataset_modeldiff_sap_en10M_sap10M_20260708_layer47_test_20260818_201459_meta.pt"
SAE_CKPT="/shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_checkpoints/modeldiff_l47/sae_sae_base_cpt_sap_16x_l47_lr1e3.pt"
GENERATION_MODEL="/fsx/ns1/pretrained/huggingface/gemma3_CPTv6_IFTv2"

FEATURES="2091 3831 3849 5286 6453 6568 7901 8746 9199 9857 11704 16840 20231 20947 22407 23626 24653 26905 28691 29622 31493 32602 33818 34237 34447 36002 36289 37672 38509 40641 86 802 1338 4245 4432 6927 10388 11615 11684 12047 13817 15097 16133 17493 23880 27011 28057 29802 30407 30571 31122 32150 32575 33373 33743 34519 35369 36630 37313 37475 37829 37848 38129 38694 39667 40495 40749 41662 41683 42065 42848 43041 43635 44514 45326 45974 46040 46282 46293 49101 49320 50561 51109 51302 51825 52583 52804 53302 53509 53970 56304 57662 59224 59272 59321 59968 60104 60295 60547 60727"

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
    --sap-only-rag \
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
    --output "/shared/home/i777599/nova_model/SAE/results/CPT-META-SAP/br3_base_expl_cptmeta_saponly_tpq20_100feat.json"

echo "Finished: $(date)"
