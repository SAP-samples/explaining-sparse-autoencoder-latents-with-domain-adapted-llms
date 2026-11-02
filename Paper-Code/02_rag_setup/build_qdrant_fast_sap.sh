#!/bin/bash
#SBATCH --job-name=qdrant_fast
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/qdrant_fast_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=60G
#SBATCH --partition=cpu
#SBATCH --constraint=c59xlarge
#SBATCH --nodelist=cpu-dy-c59xlarge-3

SCRIPT_DIR="/shared/home/i777599/nova_model/SAE/scripts/02_rag_setup"
INDEX_DIR="/shared/home/i777599/nova_model/SAE/data/RAG/vector_db_final/merged"
QDRANT_HOST="cpu-dy-c59xlarge-1"

echo "Starting fast Qdrant index build at $(date)"
echo "Index dir: $INDEX_DIR"
echo "Qdrant host: $QDRANT_HOST (gRPC port 6334)"

# Wait for Qdrant to be ready
until curl -sf "http://${QDRANT_HOST}:6333/healthz" > /dev/null 2>&1; do
    echo "  Qdrant not ready, retrying in 10s..."
    sleep 10
done
echo "Qdrant is ready."

/usr/bin/python3 "$SCRIPT_DIR/build_qdrant_index_fast.py" \
    --index-dir "$INDEX_DIR" \
    --qdrant-host "$QDRANT_HOST" \
    --grpc-port 6334 \
    --collection "sap_chunks" \
    --batch-size 8192 \
    --num-workers 16 \
    --start-from 3145728

echo "Done at $(date)"
