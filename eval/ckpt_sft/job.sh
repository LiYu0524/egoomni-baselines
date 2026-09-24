#!/usr/bin/env bash
# clusterx job: EgoAVU r100k LoRA on egoOmni shard k of K — gold protocol, then self-protocol rounds 2..4 of the same items.
# usage: job.sh K k [NPROC=8]
set -uo pipefail
K=$1; k=$2; NPROC=${3:-8}
PVC=/ai4good1-shared/liyu/egoavu; R=/ai4good1-shared/liyu/egoOmni_baselines/eval/ckpt_sft
mkdir -p $R/logs $R/configs; exec > >(tee -a $R/logs/job_s${k}of${K}_$(hostname).log) 2>&1
echo "[ckpt_sft] $(date -Is) host=$(hostname) shard=$k/$K gpus=$(nvidia-smi -L | wc -l)"
if [ ! -x /shared/egoavu/env/bin/python ]; then     # same restore as bench_infer/tools/run_bench_infer.sh
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "${avail:-0}" -ge 20 ]; then tar -xf $PVC/backup/egoavu_nvme_20260921.tar -C /
  else mkdir -p /dev/shm/egoavu_root /shared; tar -xf $PVC/backup/egoavu_nvme_20260921.tar -C /dev/shm/egoavu_root && ln -sfn /dev/shm/egoavu_root/shared/egoavu /shared/egoavu; fi
  echo "[ckpt_sft] $(date -Is) env restored"
fi
ENV=/shared/egoavu/env; LF=/shared/egoavu/vendor/LlamaFactory; PY=$ENV/bin/python
export PATH="$ENV/bin:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/tmp/ckpt_sft_cache/hf_datasets HF_HOME=/tmp/ckpt_sft_cache/hf_home WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 EGOAVU_MM_META_CACHE=1
export FORCE_TORCHRUN=1 NPROC_PER_NODE=$NPROC NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29641}
mkdir -p /tmp/ckpt_sft_cache
predict() {   # $1 = dataset name
  local ds=$1 n; n=$(wc -l < $R/data/$ds.jsonl)
  [ "$n" -eq 0 ] && { echo "[ckpt_sft] $ds empty"; return 0; }
  if [ -f $R/runs/$ds/generated_predictions.jsonl ] && [ "$(wc -l < $R/runs/$ds/generated_predictions.jsonl)" -eq "$n" ]; then echo "[ckpt_sft] $ds already predicted"
  else $PY $R/make_config.py $ds $R/configs/$ds.yaml >/dev/null; t=$(date +%s); (cd $LF && $PY -m llamafactory.cli train $R/configs/$ds.yaml) || return 1
       echo "[ckpt_sft] $(date -Is) $ds: $n rows in $(( $(date +%s) - t )) s"; fi
  (cd $R && $PY collect.py $ds $K $k)
}
cd $R
predict gold_s${k}of${K} || { echo "[ckpt_sft] gold predict FAILED"; exit 1; }
for rn in 2 3 4; do
  (cd $R && $PY build_data.py self $K $k $rn) && predict self_r${rn}_s${k}of${K} || { echo "[ckpt_sft] self round $rn FAILED"; exit 1; }
done
(cd $R && $PY collect.py none $K $k --final)
echo "[ckpt_sft] $(date -Is) DONE shard $k/$K"
