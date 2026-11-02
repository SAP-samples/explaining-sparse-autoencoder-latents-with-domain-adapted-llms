#!/bin/bash
#SBATCH --job-name=cc_webapp
#SBATCH --output=/shared/home/i777599/Webapp-Storage/logs/webapp_%j.out
#SBATCH --error=/shared/home/i777599/Webapp-Storage/logs/webapp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu-st-infranode-1

set -euo pipefail
source /shared/home/i777599/nova_venv/bin/activate

echo "Starting CrossCoder Viewer on $(hostname) — $(date)"
echo "Port: 7861"

cd /shared/home/i777599/nova_model/Interpretability-Webapp/viewer
exec python app.py --port 7861
