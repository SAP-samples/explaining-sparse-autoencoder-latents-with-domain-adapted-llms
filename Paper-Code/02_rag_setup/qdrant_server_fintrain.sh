#!/bin/bash
#SBATCH --job-name=qdrant_fintrain
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/qdrant_fintrain_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=68G
#SBATCH --partition=cpu
#SBATCH --constraint=c59xlarge

echo "Qdrant fintrain server starting on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

docker rm -f qdrant_fintrain 2>/dev/null || true

docker run --rm \
  --name qdrant_fintrain \
  -p 6335:6333 \
  -p 6336:6334 \
  -e QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=512 \
  -v /shared/home/i777599/nova_model/SAE/data/qdrant_fintrain_storage:/qdrant/storage \
  qdrant/qdrant:latest

echo "Qdrant fintrain server exited at $(date)"
