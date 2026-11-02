#!/bin/bash
#SBATCH --job-name=qdrant_server
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/qdrant_server_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=68G
#SBATCH --partition=cpu
#SBATCH --constraint=c59xlarge

echo "Qdrant server starting on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Stop any existing qdrant container
docker rm -f qdrant_sap 2>/dev/null || true

docker run --rm \
  --name qdrant_sap \
  -p 6333:6333 \
  -p 6334:6334 \
  -e QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=512 \
  -v /shared/home/i777599/nova_model/SAE/data/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

echo "Qdrant server exited at $(date)"
