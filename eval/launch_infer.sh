#!/usr/bin/env bash
# usage: launch_infer.sh <videollama2|salmonn7b|salmonn72b> [extra run_infer args, e.g. --subset subsets/dev300.json --limit 20]
# Runs gold+self protocols in one pass: 8 single-GPU workers (72B: one torchrun ZeRO-3 job over 8 GPUs). Blocks until done.
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
cd $EGO_ROOT/eval
M=$1; shift 1; EXTRA="$*"
case $M in
  videollama2) ENV=$EGO_ENVS/videollama2;  ADAPTER=videollama2;  DIR=models/VideoLLaMA2.1-7B-AV;         TAG=videollama2_7b_av; ZERO3=0 ;;
  salmonn7b)   ENV=$EGO_ENVS/salmonn2plus; ADAPTER=salmonn2plus; DIR=models/video-SALMONN2_plus_7B_full; TAG=salmonn2plus_7b;   ZERO3=0 ;;
  salmonn72b)  ENV=$EGO_ENVS/salmonn2plus; ADAPTER=salmonn2plus; DIR=models/video-SALMONN2_plus_72B_full; TAG=salmonn2plus_72b; ZERO3=1 ;;
  *) echo "unknown model $M"; exit 1 ;;
esac
NGPU=${NGPU:-8}
mkdir -p logs
if [ $ZERO3 = 1 ]; then
  echo "[$(date -Is)] $TAG: torchrun x$NGPU (ZeRO-3)"
  $ENV/bin/torchrun --nproc_per_node $NGPU --master_port ${MASTER_PORT:-29517} -m egoomni_eval.run_infer \
    --adapter $ADAPTER --model_dir $DIR --tag $TAG --zero3 $EXTRA > logs/infer_${TAG}.log 2>&1
else
  echo "[$(date -Is)] $TAG: $NGPU workers"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $ENV/bin/python -m egoomni_eval.run_infer --adapter $ADAPTER --model_dir $DIR --tag $TAG \
      --shard $g --nshards $NGPU $EXTRA > logs/infer_${TAG}_shard$g.log 2>&1 &
  done
  wait
fi
echo "[$(date -Is)] $TAG: finished"
