#!/bin/bash
#SBATCH --job-name=vllm-cpt-2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint='g524xlarge'
#SBATCH --gres=gpu:a10g:4
#SBATCH --output=/shared/home/i777599/nova_model/SAE/logs/vllm/vllm_cpt_2_%j.log
#
# vLLM endpoint for gemma3_CPTv6_IFTv2 (the domain-adapted CPT model), instance 2.
# Serves text generation AND logprobs (for Pred Acc / Detection / Fuzzing judge metrics).
#
# Submit with:  sbatch vllm_cpt_2.sh
# Endpoint URL: http://<node>:18009/v1   (see log for node name)
# Model name in API requests: "cpt-v6-iftv2"

MODEL_PATH="/fsx/ns1/pretrained/huggingface/gemma3_CPTv6_IFTv2"
SERVED_MODEL_NAME="cpt-v6-iftv2"
PORT=18009
MAX_MODEL_LEN=32768

echo "============================================================"
echo "  vLLM CPTv6_IFTv2 endpoint instance 2"
echo "  Node:  $(hostname)"
echo "  Port:  $PORT"
echo "  Model: $MODEL_PATH  (served as '$SERVED_MODEL_NAME')"
echo "  URL:   http://$(hostname):$PORT/v1"
echo "============================================================"

srun --container-image=/fsx/enroot/enroot-images/vllm/vllm_v0.27.1.sqsh \
     --container-mounts=/fsx:/fsx,/shared:/shared \
     --container-remap-root \
     --container-name=vllm_cpt_2_$USER \
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
