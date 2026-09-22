#!/usr/bin/env bash
# Gemini 3.8 Flash on the 136 evaluable restored_v3 items (API only). Budget cap $5 via cost_monitor (STOP_API).
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh; source $EGO_ROOT/.gemini_env; cd $EGO_ROOT/eval
export FFMPEG=$EGO_ENVS/minicpmo/bin/ffmpeg
export EGO_QA_PATH=/ai4good1-shared/liyu/egoOmni/test/qa_restored_v3.json EGO_CLIPS_META=$EGO_ROOT/eval/clips_meta_restored_v3.json
OUT=gemini_3_8_flash/restored_v3; N=12; rm -f STOP_API
for i in $(seq 0 $((N-1))); do
  python3 -m egoomni_eval.run_infer --adapter gemini --model_dir gemini-3.8-flash --tag gemini_3_8_flash --out_name $OUT --shard $i --nshards $N --prefetch 2 "$@" > logs/infer_gemini_restored_v3_shard$i.log 2>&1 &
done
python3 cost_monitor.py --tag $OUT --cap 5 --total_requests 136 --interval 30 > logs/cost_gemini_restored_v3.log 2>&1 &
MON=$!; wait $(jobs -p | grep -v $MON) 2>/dev/null; kill $MON 2>/dev/null
python3 cost_monitor.py --tag $OUT --cap 5 --total_requests 136 --once | tail -1
echo "[$(date -Is)] gemini restored_v3 finished"
