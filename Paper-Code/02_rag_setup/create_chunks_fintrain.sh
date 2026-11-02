#!/bin/bash
# Stage 1: Stream FinTrain finance_unsup and create shard files
# CPU-only, needs enough RAM for streaming + writing shards
#SBATCH --job-name=fintrain_chunks
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/fintrain_chunks_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/fintrain_chunks_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate

NUM_SHARDS=9
SHARD_DIR="/fsx/i777599/cache/fintrain_shards"

echo "Starting FinTrain chunking at $(date)"
echo "Output dir: $SHARD_DIR"
echo "Num shards: $NUM_SHARDS"

mkdir -p "$SHARD_DIR"

python /shared/home/i777599/nova_model/SAE/scripts/02_rag_setup/create_chunks_fintrain.py \
    --num_shards "$NUM_SHARDS" \
    --output_dir "$SHARD_DIR" \
    --chunk_size 512 \
    --chunk_overlap 64

echo "Chunking done at $(date)"
