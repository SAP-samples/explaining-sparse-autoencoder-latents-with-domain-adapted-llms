#!/bin/bash
#SBATCH --job-name=sae_acts_cpt_salesforce_20M
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/sae_acts_cpt_salesforce_20M_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/sae_acts_cpt_salesforce_20M_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10g:1
#SBATCH --mem=60G
#SBATCH --partition=gpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_activations

cd /shared/home/i777599/nova_model/SAE/scripts/05_collect_sae_activations

LOG=/shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_activations/collect_log_20M_${SLURM_JOB_ID}.txt

python collect_sae_activations.py \
    --model   "/fsx/i777599/models/Llama-Fin-8b" \
    --sae     /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_checkpoints/sae_sae_cpt_salesforce_20M_16x_l31.pt \
    --dataset /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/input_data/sae_dataset_modeldiff_salesforce_en10M_salesforce10M_20260828.jsonl \
    --layer   31 \
    --split   test \
    --batch   4 \
    --max-len 2048 \
    --out-dir /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_activations \
    2>&1 > "$LOG"
cat "$LOG"

# Extract and save the exact meta path for the downstream singlesae job
META_PATH=$(grep "Metadata:" "$LOG" | awk '{print $2}')
echo "$META_PATH" > /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_activations/last_meta_20M.txt
echo "Meta path written: $META_PATH"

echo "Done. SAE activations collected (CPT 20M, test split)."
