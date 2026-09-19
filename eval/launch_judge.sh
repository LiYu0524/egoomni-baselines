#!/usr/bin/env bash
# usage: launch_judge.sh <tag> [tag2 ...]   — local Qwen3-32B via vLLM on CUDA_VISIBLE_DEVICES (default 0,1); judges both protocols
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
cd $EGO_ROOT/eval; mkdir -p logs
export PATH=$EGO_ENVS/judge/bin:$PATH      # vLLM's compile step shells out to ninja
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
GPU_UTIL=${GPU_UTIL:-0.85}     # e.g. GPU_UTIL=0.2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 to co-run with inference workers
TP=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
for TAG in "$@"; do
  echo "[$(date -Is)] judge $TAG"
  $EGO_ENVS/judge/bin/python -m egoomni_eval.judge --tag $TAG --backend vllm --tp $TP --gpu_util $GPU_UTIL 2>&1 | tee -a logs/judge_${TAG}.log | grep -E "^\[judge\]"
done
