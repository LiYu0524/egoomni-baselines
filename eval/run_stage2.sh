#!/usr/bin/env bash
# Stage 2: after the 7B runs → retry their errors → validate ZeRO-3 phased scheduling on a mixed audio/no-audio smoke
# → pick the faster DeepSpeed config → full 72B run. Aborts (leaving STAGE2_ABORT) if the smoke fails.
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh; cd $EGO_ROOT/eval; mkdir -p logs
log() { echo "[$(date -Is)] $*"; }
wait_pid() { while kill -0 $1 2>/dev/null; do sleep 60; done; }
[ -n "${WAIT_PID:-}" ] && { log "waiting for pid $WAIT_PID (salmonn7b launcher)"; wait_pid $WAIT_PID; }
for m in videollama2 salmonn7b; do
  tag=$([ $m = videollama2 ] && echo videollama2_7b_av || echo salmonn2plus_7b)
  n=$(grep -c "\"error\": \"" preds/$tag/shard*.jsonl 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
  if [ "$n" -gt 0 ]; then log "$tag: retrying $n error rows"; ./launch_infer.sh $m --retry_errors; fi
done
python3 - <<'PY'
import json
from egoomni_eval.data import load_items, load_clips_meta, item_id
m=load_clips_meta(); its=load_items()
av=[item_id(i) for i in its if i["source_kind"]=="base_single" and m[i["clip_path"]]["has_audio"] and i["clip_duration_sec"]<60][:3]
v=[item_id(i) for i in its if i["source_kind"]=="base_single" and not m[i["clip_path"]]["has_audio"] and i["clip_duration_sec"]<60][:3]
mt=[item_id(i) for i in its if i["source_kind"]=="multi_turn" and i["round_count"]==2 and i["clip_duration_sec"]<60][:2]
json.dump({"item_ids": av+v+mt}, open("subsets/smoke8_mixed.json","w")); print("smoke8_mixed:", len(av+v+mt))
PY
best=""; best_lat=999999
for tune in 0 1; do
  rm -rf preds/smoke_72b_mixed_$tune
  log "ZeRO-3 mixed smoke, EGO_DS_TUNE=$tune"
  EGO_DS_TUNE=$([ $tune = 1 ] && echo 1 || echo "") timeout 2400 ./launch_infer.sh salmonn72b --subset subsets/smoke8_mixed.json --limit 1 --out_name smoke_72b_mixed_$tune
  read n_ok lat <<<"$(cat preds/smoke_72b_mixed_$tune/shard*.jsonl 2>/dev/null | python3 -c '
import sys,json; rows=[json.loads(l) for l in sys.stdin]; ok=[r for r in rows if r["pred"] is not None]
print(len(ok), (sum(r["latency_s"] for r in ok)/len(ok)) if ok else 999999)')"
  log "tune=$tune: ok rows=$n_ok mean latency=${lat}s"
  if [ "$n_ok" -ge 8 ] && python3 -c "import sys; sys.exit(0 if $lat < $best_lat else 1)"; then best=$tune; best_lat=$lat; fi
done
[ -z "$best" ] && { log "ZeRO-3 smoke FAILED (no config produced 8 ok rows) — aborting stage 2"; touch STAGE2_ABORT; exit 1; }
log "using EGO_DS_TUNE=$best (mean latency ${best_lat}s); launching FULL 72B"
EGO_DS_TUNE=$([ $best = 1 ] && echo 1 || echo "") ./launch_infer.sh salmonn72b
n=$(grep -c "\"error\": \"" preds/salmonn2plus_72b/shard*.jsonl 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
[ "$n" -gt 0 ] && { log "72b: retrying $n error rows"; EGO_DS_TUNE=$([ $best = 1 ] && echo 1 || echo "") ./launch_infer.sh salmonn72b --retry_errors; }
log "STAGE 2 DONE"
