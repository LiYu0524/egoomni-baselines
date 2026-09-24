#!/usr/bin/env bash
# rjob worker (started by tools/h/kit_job_h.sh, which restores the env and mounts the media): base on egoOmni shard k of K with
# the colleague's EgoToM-kit protocol (tools/h/nat_predict_lf.py in place of LLaMAFactory predict) - gold protocol, then
# self-protocol rounds 2..4 of the same items, exactly as eval/base/job.sh.   usage: job.sh K k
set -uo pipefail
K=$1; k=$2
R=/ai4good1-shared/liyu/egoOmni_baselines/eval/kit_base; M=/ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train; AD=none
mkdir -p $R/logs $R/runs; exec > >(tee -a $R/logs/job_s${k}of${K}_$(hostname).log) 2>&1
echo "[kit_base] $(date -Is) host=$(hostname) shard=$k/$K model=$M adapter=$AD"
PY=/shared/egoavu/env/bin/python
predict() {   # $1 = dataset name
  local ds=$1 n t; n=$(wc -l < $R/data/$ds.jsonl)
  [ "$n" -eq 0 ] && { echo "[kit_base] $ds empty"; return 0; }
  if [ -f $R/runs/$ds/generated_predictions.jsonl ] && [ "$(wc -l < $R/runs/$ds/generated_predictions.jsonl)" -eq "$n" ]; then echo "[kit_base] $ds already predicted"
  else t=$(date +%s); $PY /ai4good1-shared/liyu/tools/h/nat_predict_lf.py $R/data/$ds.jsonl $R/runs/$ds $M $AD 256 360 >> $R/logs/nat_$ds.log 2>&1 \
         || { tail -3 $R/logs/nat_$ds.log; return 1; }
       echo "[kit_base] $(date -Is) $ds: $n rows in $(( $(date +%s) - t )) s"; fi
  (cd $R && $PY collect.py $ds $K $k)
}
cd $R
predict gold_s${k}of${K} || { echo "[kit_base] gold predict FAILED"; exit 1; }
for rn in 2 3 4; do
  (cd $R && $PY build_data.py self $K $k $rn) && predict self_r${rn}_s${k}of${K} || { echo "[kit_base] self round $rn FAILED"; exit 1; }
done
(cd $R && $PY collect.py none $K $k --final)
echo "[kit_base] $(date -Is) DONE shard $k/$K"
