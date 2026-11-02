#!/bin/bash
#SBATCH --job-name=acts_base_salesforce_en_l31
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/acts_base_salesforce_en_l31_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/acts_base_salesforce_en_l31_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10g:1
#SBATCH --mem=60G
#SBATCH --partition=gpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/activations
mkdir -p /fsx/i777599/cache/Salesforce_modeldiff/activations

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=========================================="

cd /shared/home/i777599/nova_model/SAE/scripts/03_collect_activations

python collect_activations_salesforce.py \
    --model    "/fsx/i777599/models/Meta-Llama-3-8B-Instruct" \
    --dataset  "/shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/input_data/sae_dataset_modeldiff_salesforce_en10M_salesforce10M_20260828.jsonl" \
    --layer    31 \
    --split    train \
    --lang     en \
    --batch    1 \
    --max-len  2048 \
    --out      /shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/activations \
    --mmap-dir /fsx/i777599/cache/Salesforce_modeldiff/activations

echo "=========================================="
echo "Done. Base model (Llama-3-8B-Instruct) Layer 31 EN activations collected."
echo "Finished: $(date)"
echo "=========================================="
