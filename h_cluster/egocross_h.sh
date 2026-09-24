#!/usr/bin/env bash
# H rjob (1 H200): EgoCross closed-set (957 MCQ) for one model. Two bs=1 predict processes share the card
# (A: fps0.5 half 0 then the fps1.0 set; B: fps0.5 half 1), then official parse + scoring.   usage: egocross_h.sh TAG ADAPTER|none
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
R=/ai4good1-shared/liyu/egocross_eval_h; mkdir -p $LY/egocross_eval_h/{logs,configs,runs,results}
exec > >(tee -a $LY/egocross_eval_h/logs/job_$1_$(hostname).log) 2>&1
h_check_gpus 1 && h_cuda_home || exit 1
[ -x /shared/egoavu/env/bin/python ] || tar -xf /ai4good1-shared/liyu/egoavu/backup/egoavu_nvme_20260921.tar -C /
ENV=/shared/egoavu/env; LF=/shared/egoavu/vendor/LlamaFactory; PY=$ENV/bin/python
export PATH="$ENV/bin:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 NPROC_PER_NODE=1 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 OMP_NUM_THREADS=6
TAG=$1; AD=$2
run() {   # run DATASET FPS PORT
  local ds=$1 fps=$2 port=$3 cfg=$R/configs/${TAG}__$1.yaml n
  n=$(wc -l < $R/data/$ds.jsonl)
  if [ -f $R/runs/$TAG/$ds/generated_predictions.jsonl ] && [ "$(wc -l < $R/runs/$TAG/$ds/generated_predictions.jsonl)" -eq "$n" ]; then h_log "$ds already predicted"; return 0; fi
  $PY $R/make_config.py $TAG $AD $ds $fps $cfg
  (cd $LF && HF_DATASETS_CACHE=/tmp/egx_$ds/hf HF_HOME=/tmp/egx_$ds/home MASTER_PORT=$port $PY -m llamafactory.cli train $cfg) > $R/logs/${TAG}__${ds}.log 2>&1 \
    || { h_log "$ds FAILED (see logs/${TAG}__${ds}.log)"; return 1; }
  h_log "$ds done ($n questions)"
}
(run egocross_fps05_s0of2 0.5 29640 && run egocross_fps10 1.0 29660) & A=$!
run egocross_fps05_s1of2 0.5 29650 & B=$!
rc=0; wait $A || rc=1; wait $B || rc=1
[ $rc -eq 0 ] && $PY $R/collect.py $TAG || { h_log "EgoCross $TAG FAILED"; exit 1; }
h_log "EgoCross $TAG DONE"
