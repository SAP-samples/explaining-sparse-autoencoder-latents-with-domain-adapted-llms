#!/bin/bash
# Upload FinTrain vector DB to Qdrant (collection: fintrain_chunks)
#SBATCH --job-name=fintrain_qdrant
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/fintrain_qdrant_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --constraint=c524xlarge

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate

INDEX_DIR="/shared/home/i777599/nova_model/SAE/data/RAG/fintrain_vector_db/merged"
QDRANT_HOST="cpu-dy-c59xlarge-3"
QDRANT_URL="http://${QDRANT_HOST}:6335"
QDRANT_GRPC_PORT=6336

echo "Starting FinTrain Qdrant upload at $(date)"
echo "Index dir: $INDEX_DIR"
echo "Qdrant host: $QDRANT_HOST (REST :6335, gRPC :6336)"

echo "Waiting for Qdrant to be ready..."
until curl -sf "$QDRANT_URL/healthz" > /dev/null 2>&1; do
    echo "  Qdrant not ready yet, retrying in 10s..."
    sleep 10
done
echo "Qdrant is ready."

python /shared/home/i777599/nova_model/SAE/scripts/02_rag_setup/build_qdrant_index_fast.py \
    --index-dir  "$INDEX_DIR" \
    --qdrant-host "$QDRANT_HOST" \
    --grpc-port "$QDRANT_GRPC_PORT" \
    --collection fintrain_chunks \
    --batch-size 8192 \
    --num-workers 16 \
    --start-from 0 \
    --recreate

echo "Upload done at $(date)"
