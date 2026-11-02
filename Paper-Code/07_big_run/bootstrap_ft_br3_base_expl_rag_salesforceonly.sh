#!/bin/bash
#SBATCH --job-name=ft-br3-base-expl-rag-salesforceonly
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint='g524xlarge'
#SBATCH --gres=gpu:a10g:2
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/ft_br3_base_expl_rag_salesforceonly_%j.log
#
# Fintrain | branching=1 | BASE+RAG explainer | CPT + BASE+RAG judges | 100 features | domain-only tokens

CPT_NODE1="gpu-dy-g524xlarge-12"
BASE_NODE1="gpu-dy-g524xlarge-11"
EXPL_NODE="gpu-dy-g524xlarge-11"
QDRANT_URL="http://cpu-dy-c59xlarge-3:6335"

META="/shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_activations/sae_acts_Llama-Fin-8b_sae_sae_cpt_salesforce_20M_16x_l31_sae_dataset_modeldiff_salesforce_en10M_salesforce10M_20260828_layer31_test_20260831_094209_meta.pt"
SAE_CKPT="/shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_checkpoints/sae_sae_cpt_salesforce_20M_16x_l31.pt"
GENERATION_MODEL="/fsx/i777599/models/Llama-Fin-8b"

FEATURES="64751 2626 6696 7936 23757 24032 25414 38063 39302 42407 42939 48199 48581 54488 57338 9498 23183 36900 37840 45549 54827 61366 61410 330 3102 3686 3705 4007 5617 6485 7059 8469 8570 9844 10480 10559 12093 12613 13018 15554 15812 16105 17763 18720 19306 21530 23299 29024 29037 29096 30303 31128 33768 34475 34742 35040 35373 36882 37226 37707 39470 40026 40861 41318 45392 47815 48317 49148 49159 49780 49912 50115 50740 53771 58334 60601 63743 64256 65459 534 844 1386 1627 2324 6410 10434 10767 22396 23525 27407 29809 30658 31753 35393 35750 37304 37663 38806 38878 39277"

echo "============================================================"
echo "  Fintrain br1 | BASE+RAG explainer | CPT + BASE+RAG judges | 100 features | domain-only tokens"
echo "  Node: $(hostname)  Started: $(date)"
echo "============================================================"

source /shared/home/i777599/nova_venv/bin/activate

python /shared/home/i777599/nova_model/SAE/scripts/paper_vllm_confidence_interval_fintrain.py \
    --features $FEATURES \
    --meta "$META" \
    --sae-ckpt "$SAE_CKPT" \
    --generation-model "$GENERATION_MODEL" \
    --explainer-endpoint "http://${EXPL_NODE}:18004/v1" \
    --explainer-served-model "llama-3-8b-instruct" \
    --explainer-rag \
    --sap-only-rag \
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
    --output "/shared/home/i777599/nova_model/SAE/results/ft_br3_base_expl_rag_salesforceonly_baserag_100feat.json"

echo "Finished: $(date)"
