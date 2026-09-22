#!/usr/bin/env bash
# Dev-box half of the restored_v3 run: waits for RESTORED_V3_READY, runs the three 7B models on the local 8 GPUs, then
# takes over any 72B slice whose clusterx job has still not started (stops that job first so nothing writes twice).
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh; cd $EGO_ROOT/eval; mkdir -p logs
C=/root/.venvs/clusterx/bin/clusterx
log() { echo "[$(date -Is)] $*"; }
until [ -f RESTORED_V3_READY ]; do sleep 20; done
export EGO_QA_PATH=/ai4good1-shared/liyu/egoOmni/test/qa_restored_v3.json
export EGO_CLIPS_META=/ai4good1-shared/liyu/egoOmni_baselines/eval/clips_meta_restored_v3.json
export LOG_SUFFIX=_restored_v3_devbox
log "items: $(python3 -c "import json; print(len(json.load(open('$EGO_QA_PATH'))))")"
for m in videollama2:videollama2_7b_av salmonn7b:salmonn2plus_7b minicpmo:minicpmo_2_6_8b; do
  model=${m%%:*}; tag=${m##*:}; log "== $model"
  if [ $model = minicpmo ]; then PATH=$EGO_ENVS/minicpmo/bin:$PATH ./launch_infer.sh $model --out_name $tag/restored_v3; else ./launch_infer.sh $model --out_name $tag/restored_v3; fi
  e=$(grep -c '"error": "' preds/$tag/restored_v3/shard*.jsonl 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
  log "$tag: rows=$(cat preds/$tag/restored_v3/shard*.jsonl 2>/dev/null | wc -l) errors=$e"
  [ "$e" -gt 0 ] && ./launch_infer.sh $model --out_name $tag/restored_v3 --retry_errors
done
for k in 0 1 2 3; do
  st=$(COLUMNS=200 timeout 60 $C list 2>/dev/null | grep "egoomni-s72b-rv3-p$k " | grep -oE "JobStatus\.[A-Z]+")
  if [ "$st" = "JobStatus.QUEU" ] || [ "$st" = "JobStatus.QUEUING" ] || { [ -z "$st" ] && [ ! -d preds/salmonn2plus_72b/restored_v3/p$k ]; }; then
    log "72B part p$k still not started ($st) → stopping job and running it here"
    timeout 60 $C stop egoomni-s72b-rv3-p$k >/dev/null 2>&1
    EGO_DS_TUNE=1 LOG_SUFFIX=_restored_v3_p$k ./launch_infer.sh salmonn72b --subset subsets/restored_v3_part${k}of4.json --out_name salmonn2plus_72b/restored_v3/p$k
  fi
done
log "DEVBOX DONE"
