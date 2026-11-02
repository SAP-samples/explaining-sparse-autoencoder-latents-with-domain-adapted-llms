#!/bin/bash
#SBATCH --job-name=merge_vecdb
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/merge_vecdb_%j.out
#SBATCH --error=/shared/home/i777599/nova_model/SAE/logs/merge_vecdb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate

cd /shared/home/i777599/nova_model/SAE/scripts/02_rag_setup

python merge_vector_db.py \
    --partial_dir /shared/home/i777599/nova_model/SAE/data/RAG/vector_db_final/partials \
    --output_dir  /shared/home/i777599/nova_model/SAE/data/RAG/vector_db_final/merged
