#!/usr/bin/env bash
# usage: launch_api.sh <model, e.g. gemini-3.8-flash> [n_workers=12] [extra run_infer args]   — API workers (no GPU) + cost monitor
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh; source $EGO_ROOT/.gemini_env; cd $EGO_ROOT/eval; mkdir -p logs
export FFMPEG=$EGO_ENVS/minicpmo/bin/ffmpeg      # libx264 build
MODEL=$1; N=${2:-12}; shift; shift || true; EXTRA="$*"
TAG=$(echo $MODEL | tr ".-" "__"); CAP=${CAP:-100}
rm -f STOP_API
echo "[$(date -Is)] $TAG: $N API workers, cap \$$CAP"
for i in $(seq 0 $((N-1))); do
  python3 -m egoomni_eval.run_infer --adapter gemini --model_dir $MODEL --tag $TAG --shard $i --nshards $N --prefetch 2 $EXTRA > logs/infer_${TAG}_shard$i.log 2>&1 &
done
python3 cost_monitor.py --tag $TAG --cap $CAP > logs/cost_${TAG}.log 2>&1 &
MON=$!
wait $(jobs -p | grep -v $MON) 2>/dev/null; kill $MON 2>/dev/null
python3 cost_monitor.py --tag $TAG --cap $CAP --once | tail -1
echo "[$(date -Is)] $TAG: finished"
