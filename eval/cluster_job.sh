#!/usr/bin/env bash
# clusterx job body: restore the model's runtime env from its PVC tarball onto the node's local disk (same path as on the
# dev box, so conda prefixes/shebangs stay valid), then run one inference slice of the restored_v3 items.
# usage (inside the job): cluster_job.sh <videollama2|salmonn7b|salmonn72b|minicpmo> <part_tag> [launch_infer args...]
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
M=$1; PART=$2; shift 2
case $M in videollama2) E=videollama2 ;; salmonn7b|salmonn72b) E=salmonn2plus ;; minicpmo) E=minicpmo ;; *) echo "bad model $M"; exit 2 ;; esac
echo "[$(date -Is)] host=$(hostname) gpus=$(nvidia-smi -L 2>/dev/null | wc -l) model=$M part=$PART args=$*"
if [ ! -x $EGO_ENVS/$E/bin/python ]; then
  mkdir -p $EGO_ENVS; t=$(date +%s); tar -C $EGO_ENVS -xf $EGO_ROOT/envs/$E.tar || { echo "env restore failed"; exit 3; }
  echo "[$(date -Is)] env $E restored in $(( $(date +%s) - t ))s"
fi
# the restored_v3 media/QA are prepared by another session; wait for the go marker (written once they are final)
w=0; until [ -f $EGO_ROOT/eval/RESTORED_V3_READY ]; do [ $w = 0 ] && echo "[$(date -Is)] waiting for RESTORED_V3_READY"; w=1; sleep 20; done
export EGO_QA_PATH=/ai4good1-shared/liyu/egoOmni/test/qa_restored_v3.json
export EGO_CLIPS_META=/ai4good1-shared/liyu/egoOmni_baselines/eval/clips_meta_restored_v3.json
export LOG_SUFFIX=_restored_v3_$PART
[ $M = salmonn72b ] && export EGO_DS_TUNE=1
[ $M = minicpmo ] && export PATH=$EGO_ENVS/minicpmo/bin:$PATH
cd $EGO_ROOT/eval
./launch_infer.sh $M "$@"
echo "[$(date -Is)] job done rc=$?"
