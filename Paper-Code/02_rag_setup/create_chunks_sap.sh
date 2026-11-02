#!/bin/bash
#SBATCH --job-name=create_chunks
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/create_chunks_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/create_chunks_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate

cd /shared/home/i777599/nova_model/SAE/scripts/02_rag_setup

python create_chunks_sap.py \
    --num_shards 9 \
    --output_dir /fsx/i777599/cache/vecdb_shards
