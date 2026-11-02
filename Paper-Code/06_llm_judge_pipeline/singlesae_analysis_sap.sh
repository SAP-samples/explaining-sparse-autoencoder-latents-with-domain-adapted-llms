#!/bin/bash
#SBATCH --job-name=ssae_saeBASE_modelCPT
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/ssae_saeBASE_modelCPT_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/ssae_saeBASE_modelCPT_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10g:1
#SBATCH --mem=60G
#SBATCH --partition=gpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate
export PYTHONUNBUFFERED=1

ACTS_DIR=/shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_activations
CKPT_DIR=/shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_checkpoints/modeldiff_l47
OUT_DIR=/shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/singlesae_results
mkdir -p "$OUT_DIR"
cd /shared/home/i777599/nova_model/SAE/scripts/06_llm_judge_pipeline

TAG=saeBASE_modelCPT
SAE_PT="$CKPT_DIR/sae_sae_base_cpt_sap_16x_l47_lr1e3.pt"
META_GLOB="gemma3_CPTv6_sae_sae_base_cpt_sap_16x_l47_lr1e3"

# Newest matching meta.pt from the corresponding collection run (dependency guarantees it exists)
META_PT=$(ls -t "$ACTS_DIR"/sae_acts_${META_GLOB}_*_layer47_test_*_meta.pt 2>/dev/null | head -1)
if [[ -z "$META_PT" ]]; then
  echo "ERROR: no meta.pt found for glob '$META_GLOB' in $ACTS_DIR"; exit 1
fi

echo "TAG=$TAG"
echo "META=$META_PT"
echo "SAE=$SAE_PT"

python singlesae_analysis_sap.py \
    --meta                "$META_PT" \
    --sae                 "$SAE_PT" \
    --n-quantiles         10 \
    --tokens-per-quantile 100 \
    --context-window      10 \
    --workers             8 \
    --checkpoint-every    500 \
    --tag                 "$TAG" \
    --out-dir             "$OUT_DIR"

echo "Done. single-SAE analysis ($TAG) complete."
