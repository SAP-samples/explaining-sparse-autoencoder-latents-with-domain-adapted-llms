#!/bin/bash
#SBATCH --job-name=acts_base_en_l47
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/acts_base_en_l47_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/acts_base_en_l47_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10g:1
#SBATCH --mem=60G
#SBATCH --partition=gpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/activations
mkdir -p /fsx/i777599/cache/SAP_modeldiff/activations

cd /shared/home/i777599/nova_model/SAE/scripts/03_collect_activations

python collect_activations_sap.py \
    --model    "google/gemma-3-12b-pt" \
    --dataset  "/shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/input_data/sae_dataset_modeldiff_sap_en10M_sap10M_20260708.jsonl" \
    --layer    47 \
    --split    train \
    --lang     en \
    --batch    1 \
    --max-len  2048 \
    --out      /shared/home/i777599/nova_model/SAE/data/SAP_modeldiff/activations \
    --mmap-dir /fsx/i777599/cache/SAP_modeldiff/activations

echo "Done. Base model Layer 47 EN activations collected."
