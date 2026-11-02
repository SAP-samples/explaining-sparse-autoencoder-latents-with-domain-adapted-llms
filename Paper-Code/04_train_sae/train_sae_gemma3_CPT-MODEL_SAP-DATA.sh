#!/bin/bash
#SBATCH --job-name=sae_cpt_sap_16x_l47
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/sae_cpt_sap_16x_l47_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/sae_cpt_sap_16x_l47_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10g:1
#SBATCH --mem=60G
#SBATCH --partition=gpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_checkpoints/modeldiff_l47

cd /shared/home/i777599/nova_model/SAE/scripts/04_train_sae

python train_sae_gemma3.py \
    --activations /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/activations/acts_gemma3_CPTv6_sae_dataset_modeldiff_sap_en10M_sap10M_20260708_layer47_train_sap_20260728.pt \
    --n-features      61440 \
    --l0-target       150 \
    --lr              1e-3 \
    --freq-lambda     0 \
    --sparsity-lambda 1e-4 \
    --kde-bandwidth   0.001 \
    --epochs          10 \
    --batch           4096 \
    --normalize \
    --warmup-frac     0.1 \
    --theta-init      0.001 \
    --init-from       /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_checkpoints/modeldiff_l47/sae_sae_base_16x_l47_lr1e3.pt \
    --out-dir         /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_checkpoints/modeldiff_l47 \
    --sae-name        sae_base_cpt_sap_16x_l47_lr1e3

echo "Done. BASE-init CPT+SAP 16x SAE complete."
