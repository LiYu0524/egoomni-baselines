#!/usr/bin/env bash
# H rjob (1 H200): rebalanced tail of the kit-protocol EgoAVU-Bench base run — nat_predict_lf.py on rest shards kitrest_s{j}of{R}
# (split_kit_rest.py), one process per shard on this card; then finalize_kit_rest.py (the last job to finish merges + collects).
# usage: kit_eab_rest_h.sh R SHARD [SHARD ...]
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
R=$1; shift
mkdir -p $LY/natkit/logs; exec > >(tee -a $LY/natkit/logs/kitrest_$(hostname).log) 2>&1
h_check_gpus 1 && h_mount_s3 || exit 1
[ -x /shared/egoavu/env/bin/python ] || tar -xf /ai4good1-shared/liyu/egoavu/backup/egoavu_nvme_20260921.tar -C /
FFB=/ai4good1-shared/liyu/egoOmni_baselines/envs_h/ffmpeg/bin
export PATH="/shared/egoavu/env/bin:$FFB:$PATH" PYTHONPATH=/ai4good1-shared/liyu/tools/nat/site FFPROBE=$FFB/ffprobe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NAT_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export FORCE_QWENVL_VIDEO_READER=decord PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
BI=/ai4good1-shared/liyu/egoavu/bench_infer; PY=/shared/egoavu/env/bin/python; M=/ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train
h_log "kit rest shards $* of $R"
pids=()
for j in "$@"; do
  $PY /ai4good1-shared/liyu/tools/h/nat_predict_lf.py $BI/data/kitrest_s${j}of${R}.jsonl $BI/runs/kit_base_rest/kitrest_s${j}of${R} $M none 1024 360 \
    >> $BI/logs/kit_base_rest_s${j}of${R}.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
(cd /tmp && $PY /ai4good1-shared/liyu/tools/h/finalize_kit_rest.py $R) || rc=1
h_log "kit rest $* of $R exit $rc"; exit $rc
