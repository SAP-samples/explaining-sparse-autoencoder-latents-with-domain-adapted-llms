#!/bin/bash
#SBATCH --job-name=acts_saeBASE_modelCPT
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/acts_saeBASE_modelCPT_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/acts_saeBASE_modelCPT_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10g:1
#SBATCH --mem=60G
#SBATCH --partition=gpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_activations

cd /shared/home/i777599/nova_model/SAE/scripts/05_collect_sae_activations

python collect_sae_activations.py \
    --model   "/fsx/i777599/models/gemma3_CPTv6" \
    --sae     /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_checkpoints/modeldiff_l47/sae_sae_base_cpt_sap_16x_l47_lr1e3.pt \
    --dataset /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/input_data/sae_dataset_modeldiff_sap_en10M_sap10M_20260708.jsonl \
    --layer   47 \
    --split   test \
    --batch   1 \
    --max-len 2048 \
    --out-dir /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/sae_activations

echo "Done. Acts: SAE=BASE(29059) x Model=CPT(gemma3_CPTv6)"
