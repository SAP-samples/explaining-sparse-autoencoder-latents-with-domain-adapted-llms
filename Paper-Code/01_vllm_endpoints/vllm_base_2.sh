#!/bin/bash
#SBATCH --job-name=vllm-base-2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint='g524xlarge'
#SBATCH --gres=gpu:a10g:4
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/vllm/vllm_base_2_%j.log
#
# vLLM endpoint for the BASE model (google/gemma-3-12b-it), instance 2.
# Used as the Explainer / generation model for the BASE and BASE+RAG runs.
#
# Submit with:  sbatch vllm_base_2.sh
# Endpoint URL: http://<node>:18008/v1   (see log for node name)
# Model name in API requests: "gemma-3-12b-it"

MODEL_PATH="/fsx/i777599/models/gemma-3-12b-it-textfix"
SERVED_MODEL_NAME="gemma-3-12b-it"
PORT=18008
MAX_MODEL_LEN=32768

echo "============================================================"
echo "  vLLM BASE (gemma-3-12b-it) endpoint instance 2"
echo "  Node:  $(hostname)"
echo "  Port:  $PORT"
echo "  Model: $MODEL_PATH  (served as '$SERVED_MODEL_NAME')"
echo "  URL:   http://$(hostname):$PORT/v1"
echo "============================================================"

srun --container-image=/fsx/enroot/enroot-images/vllm/vllm_v0.27.1.sqsh \
     --container-mounts=/fsx:/fsx,/shared:/shared \
     --container-remap-root \
     --container-name=vllm_base_2_$USER \
       vllm serve "$MODEL_PATH" \
         --host 0.0.0.0 \
         --port $PORT \
         --tensor-parallel-size 4 \
         --max-model-len $MAX_MODEL_LEN \
         --served-model-name "$SERVED_MODEL_NAME" \
         --gpu-memory-utilization 0.85 \
         --max-num-seqs 32 \
         --enable-chunked-prefill \
         --trust-remote-code
