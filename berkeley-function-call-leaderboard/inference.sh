#!/bin/bash
#SBATCH --job-name=0047_evaluation
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out


source .venv/bin/activate

MODEL_NAME=$1
MODEL_PATH=$2


bfcl generate \
  --model ${MODEL_NAME} \
  --backend vllm \
  --num-gpus 4 \
  --gpu-memory-utilization 0.9 \
  --local-model-path ${MODEL_PATH}   # ← optional

bfcl evaluate \
  --model ${MODEL_NAME}
