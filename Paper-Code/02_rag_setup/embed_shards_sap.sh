#!/bin/bash
#SBATCH --job-name=embed_shards
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/embed_shards_%A_%a.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/embed_shards_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a10g:1
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --array=0-8

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SHARD_DIR="/fsx/i777599/cache/vecdb_shards"
PARTIAL_DIR="/shared/home/i777599/nova_model/SAE/data/RAG/vector_db_final/partials"

SHARD_PATH="${SHARD_DIR}/$(printf 'shard_%04d.jsonl' $SLURM_ARRAY_TASK_ID)"
OUTPUT_DIR="${PARTIAL_DIR}/$(printf '%04d' $SLURM_ARRAY_TASK_ID)"

echo "Task ID    : $SLURM_ARRAY_TASK_ID"
echo "Shard      : $SHARD_PATH"
echo "Output     : $OUTPUT_DIR"

cd /shared/home/i777599/nova_model/SAE/scripts/02_rag_setup

python embed_shard.py \
    --shard_path "$SHARD_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 128
