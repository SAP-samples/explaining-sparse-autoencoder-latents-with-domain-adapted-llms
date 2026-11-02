#!/bin/bash
#SBATCH --job-name=sae_base_salesforce_20M_16x_l31
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/sae_base_salesforce_20M_16x_l31_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/sae_base_salesforce_20M_16x_l31_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10g:1
#SBATCH --mem=60G
#SBATCH --partition=gpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_checkpoints

cd /shared/home/i777599/nova_model/SAE/scripts/04_train_sae

python train_sae_llama3.py \
    --activations /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/activations/acts_Meta-Llama-3-8B-Instruct_sae_dataset_modeldiff_salesforce_en10M_salesforce10M_20260828_layer31_train_en_20260828.pt \
    --n-features      65536 \
    --l0-target       150 \
    --lr              1e-3 \  # 2e-4
    --freq-lambda     0 \
    --sparsity-lambda 1e-4 \  # 2e-4
    --kde-bandwidth   0.001 \
    --epochs          10 \
    --batch           4096 \  # 2048
    --normalize \
    --warmup-frac     0.1 \
    --theta-init      0.001 \
    --out-dir         /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/sae_checkpoints \
    --sae-name        sae_base_salesforce_20M_16x_l31

echo "Done. BASE 20M SAE training complete."
