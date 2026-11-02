#!/bin/bash
#SBATCH --job-name=vllm-cpt-ft1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint='g524xlarge'
#SBATCH --gres=gpu:a10g:4
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/vllm/vllm_cpt_fintrain_1_%j.log
#
# vLLM endpoint for the CPT Fintrain model (Llama-Fin-8b), instance 1.
# Endpoint URL: http://<node>:18006/v1
# Model name in API requests: "llama-fin-8b"

MODEL_PATH="/fsx/i777599/models/Llama-Fin-8b"
SERVED_MODEL_NAME="llama-fin-8b"
PORT=18006
MAX_MODEL_LEN=8192

echo "============================================================"
echo "  vLLM CPT-Fintrain (Llama-Fin-8b) instance 1"
echo "  Node:  $(hostname)"
echo "  Port:  $PORT"
echo "  Model: $MODEL_PATH  (served as '$SERVED_MODEL_NAME')"
echo "  URL:   http://$(hostname):$PORT/v1"
echo "============================================================"

srun --container-image=/fsx/enroot/enroot-images/vllm/vllm_v0.27.1.sqsh \
     --container-mounts=/fsx:/fsx,/shared:/shared \
     --container-remap-root \
     --container-name=vllm_cpt_ft1_$USER \
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
