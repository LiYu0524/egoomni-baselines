#!/usr/bin/env bash
# EgoAVU-Bench inference job (clusterx or dev box). usage: run_bench_infer.sh TAG ADAPTER_DIR|none DATASET [NPROC]
# Restores the EgoAVU env (NVMe tarball) if this container lacks it, then runs LLaMAFactory predict and collects.
set -euo pipefail
TAG="${1:?tag}"; ADAPTER="${2:?adapter dir or none}"; DATASET="${3:?dataset}"; NPROC="${4:-8}"
PVC=/ai4good1-shared/liyu/egoavu; BI=$PVC/bench_infer
HOST="$(hostname)"; RUN="${TAG}__${DATASET}"
LOG=$BI/logs/${RUN}_${HOST}.log; mkdir -p $BI/logs $BI/configs $BI/telemetry
exec > >(tee -a "$LOG") 2>&1
echo "[bench] $(date -Is) host=$HOST tag=$TAG adapter=$ADAPTER dataset=$DATASET nproc=$NPROC"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
df -h / /dev/shm 2>/dev/null | tail -2

if [ ! -x /shared/egoavu/env/bin/python ]; then
  # prefer local disk at /shared/egoavu; fall back to RAM-backed /dev/shm + symlink (venv shebangs are absolute)
  avail_root=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "${avail_root:-0}" -ge 20 ]; then
    echo "[bench] restoring env tarball to / (root has ${avail_root}G free)"; tar -xf $PVC/backup/egoavu_nvme_20260921.tar -C /
  else
    echo "[bench] root has only ${avail_root}G; restoring env into /dev/shm"; mkdir -p /dev/shm/egoavu_root /shared
    tar -xf $PVC/backup/egoavu_nvme_20260921.tar -C /dev/shm/egoavu_root && ln -sfn /dev/shm/egoavu_root/shared/egoavu /shared/egoavu
  fi
fi
ENV=/shared/egoavu/env; LF=/shared/egoavu/vendor/LlamaFactory
$ENV/bin/python -c "import llamafactory, torch, flash_attn, peft; from llamafactory.data.mm_plugin import _egoavu_split_fragment; print('[bench] env ok', torch.__version__, flash_attn.__version__, peft.__version__, torch.cuda.device_count(), 'gpus')"

CFG=$BI/configs/${RUN}.yaml
$ENV/bin/python $BI/tools/make_config.py "$TAG" "$ADAPTER" "$DATASET" "$CFG"
export PATH="$ENV/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/tmp/egoavu_bench_cache/hf_datasets HF_HOME=/tmp/egoavu_bench_cache/hf_home
export WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export FORCE_TORCHRUN=1 NPROC_PER_NODE="$NPROC" NNODES=1 NODE_RANK=0
export MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29640}"
export EGOAVU_MM_META_CACHE=1
mkdir -p /tmp/egoavu_bench_cache
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,power.limit --format=csv -l 30 \
  > "$BI/telemetry/${RUN}_${HOST}.csv" 2>&1 &
TEL=$!; trap 'kill "$TEL" 2>/dev/null || true' EXIT
cd "$LF"
t0=$(date +%s)
$ENV/bin/python -m llamafactory.cli train "$CFG"
echo "[bench] predict finished in $(( $(date +%s) - t0 )) s"
$ENV/bin/python $BI/tools/collect_predictions.py "$TAG" "$DATASET"
echo "[bench] $(date -Is) DONE $RUN"
