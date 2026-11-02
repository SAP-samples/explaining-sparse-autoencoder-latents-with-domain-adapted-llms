#!/bin/bash
# Stage 3: Merge all partial FinTrain indexes into final DB
#SBATCH --job-name=fintrain_merge
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/fintrain_merge_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/fintrain_merge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=cpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate

PARTIAL_DIR="/shared/home/i777599/nova_model/SAE/data/RAG/fintrain_vector_db/partials"
OUTPUT_DIR="/shared/home/i777599/nova_model/SAE/data/RAG/fintrain_vector_db/merged"

echo "Starting merge at $(date)"
echo "Partial dir : $PARTIAL_DIR"
echo "Output dir  : $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

python /shared/home/i777599/nova_model/SAE/scripts/02_rag_setup/merge_vector_db.py \
    --partial_dir "$PARTIAL_DIR" \
    --output_dir  "$OUTPUT_DIR"

echo "Merge done at $(date)"
