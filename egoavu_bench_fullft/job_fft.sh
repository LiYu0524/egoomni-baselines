#!/usr/bin/env bash
# clusterx job: full-FT Qwen2.5-Omni thinker on one shard of EgoAVU-Bench. usage: job_fft.sh TAG MODEL_DIR DATASET [NPROC]
set -uo pipefail
TAG=$1; MODEL=$2; DATASET=$3; NPROC=${4:-8}
PVC=/ai4good1-shared/liyu/egoavu; R=$PVC/bench_infer_fft
mkdir -p $R/logs $R/configs $R/runs; exec > >(tee -a $R/logs/${TAG}__${DATASET}_$(hostname).log) 2>&1
echo "[fft] $(date -Is) host=$(hostname) tag=$TAG dataset=$DATASET gpus=$(nvidia-smi -L | wc -l)"
until [ -f $R/MODEL_READY ]; do echo "[fft] waiting for MODEL_READY"; sleep 30; done
if [ ! -x /shared/egoavu/env/bin/python ]; then
  avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "${avail:-0}" -ge 20 ]; then tar -xf $PVC/backup/egoavu_nvme_20260921.tar -C /
  else mkdir -p /dev/shm/egoavu_root /shared; tar -xf $PVC/backup/egoavu_nvme_20260921.tar -C /dev/shm/egoavu_root && ln -sfn /dev/shm/egoavu_root/shared/egoavu /shared/egoavu; fi
  echo "[fft] $(date -Is) env restored"
fi
ENV=/shared/egoavu/env; LF=/shared/egoavu/vendor/LlamaFactory; PY=$ENV/bin/python
export PATH="$ENV/bin:$PATH" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/tmp/fft_cache/hf_datasets HF_HOME=/tmp/fft_cache/hf_home WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 EGOAVU_MM_META_CACHE=1
export FORCE_TORCHRUN=1 NPROC_PER_NODE=$NPROC NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29660
mkdir -p /tmp/fft_cache
CFG=$R/configs/${TAG}__${DATASET}.yaml
$PY $R/make_config_fft.py "$TAG" "$MODEL" "$DATASET" "$CFG" >/dev/null
t0=$(date +%s); (cd $LF && $PY -m llamafactory.cli train "$CFG") || { echo "[fft] predict FAILED"; exit 1; }
echo "[fft] $(date -Is) predict done in $(( $(date +%s) - t0 )) s"
$PY $R/collect_fft.py "$TAG" "$DATASET"
echo "[fft] $(date -Is) DONE $TAG $DATASET"
