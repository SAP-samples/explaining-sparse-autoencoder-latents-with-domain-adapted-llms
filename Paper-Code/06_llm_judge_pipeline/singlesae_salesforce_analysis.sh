#!/bin/bash
#SBATCH --job-name=singlesae_salesforce_20M
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/singlesae_salesforce_20M_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/singlesae_salesforce_20M_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10g:1
#SBATCH --mem=80G
#SBATCH --partition=gpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate

mkdir -p /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/singlesae_results

cd /shared/home/i777599/nova_model/SAE/scripts/06_llm_judge_pipeline

# Read the exact meta path written by the upstream collect job
META_FILE=/shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_activations/last_meta_20M.txt
if [ ! -f "$META_FILE" ]; then
    echo "ERROR: $META_FILE not found — upstream collect job may have failed"
    exit 1
fi
META=$(cat "$META_FILE")
if [ -z "$META" ] || [ ! -f "$META" ]; then
    echo "ERROR: Meta file not found or empty: $META"
    exit 1
fi

echo "Using meta: $META"

python singlesae_salesforce_analysis.py \
    --meta    "$META" \
    --sae     /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_checkpoints/sae_sae_cpt_salesforce_20M_16x_l31.pt \
    --split   test \
    --out-dir /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/singlesae_results \
    --tag     cpt_20M \
    --workers 8 \
    --checkpoint-every 500

echo "Done. SingleSAE Salesforce analysis complete (20M)."
