#!/bin/bash
#SBATCH --job-name=gen_dataset_salesforce
#SBATCH --exclude=gpu-dy-g516xlarge-1,gpu-dy-g548xlarge-1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/gen_dataset_salesforce_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/gen_dataset_salesforce_%j.err

set -euo pipefail

source /shared/home/i777599/nova_venv/bin/activate

OUTDIR="/shared/home/i777599/nova_model/SAE/data/Salesforce_modeldiff/input_data"
mkdir -p "$OUTDIR"

DATE=$(date +%Y%m%d)
OUT="$OUTDIR/sae_dataset_modeldiff_salesforce_en10M_salesforce10M_${DATE}.jsonl"

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"
echo "Output:  $OUT"
echo "=========================================="

cd /shared/home/i777599/nova_model/SAE/scripts

python3 generate_dataset_modeldiff_salesforce.py \
    --total-tokens 20_000_000 \
    --chunk-tokens 2048 \
    --seed 42 \
    --hf-shuffle-buffer 10_000 \
    --out "$OUT"

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
