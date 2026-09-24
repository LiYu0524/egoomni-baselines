#!/usr/bin/env bash
# H rjob (1 H200): EgoToM-kit-protocol native inference (tools/h/nat_infer.py) of one model on one benchmark — one process per
# given shard, all on this card (bs=1 decode leaves most of an H200 idle, as in the LLaMAFactory jobs).
# usage: nat_job_h.sh MODEL BENCH NUM_SHARDS SHARD [SHARD ...]      MODEL = base | ckpt_epoch2 | ckpt_fft_epoch2
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
MODEL=$1; BENCH=$2; NS=$3; shift 3
K=/ai4good1-shared/liyu/natkit; O=$K/out/$BENCH/$MODEL; mkdir -p $LY/natkit/logs $O
exec > >(tee -a $LY/natkit/logs/job_${BENCH}_${MODEL}_$(hostname).log) 2>&1
h_check_gpus 1 || exit 1
[ -x /shared/egoavu/env/bin/python ] || tar -xf /ai4good1-shared/liyu/egoavu/backup/egoavu_nvme_20260921.tar -C /
case $MODEL in
  base)            M=/ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train; AD=none ;;
  ckpt_epoch2)     M=/ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train; AD=/ai4good1-shared/liyu/egoavu/bench_infer/adapters/ckpt_epoch2 ;;
  ckpt_fft_epoch2) M=/ai4good1-shared/liyu/egoavu/models/groo_ckpt_fft_epoch2; AD=none ;;
  *) h_log "unknown model $MODEL"; exit 2 ;;
esac
FFB=/ai4good1-shared/liyu/egoOmni_baselines/envs_h/ffmpeg/bin   # ffprobe (audio check) + ffmpeg (audioread decodes the video's audio)
export PATH="/shared/egoavu/env/bin:$FFB:$PATH" PYTHONPATH=/ai4good1-shared/liyu/tools/nat/site FFPROBE=$FFB/ffprobe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NAT_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2   # 4 processes share the 8 CPUs of a 1-GPU share (kit: 4 threads, 1 process/GPU)
export FORCE_QWENVL_VIDEO_READER=decord PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
h_log "nat $BENCH $MODEL shards $* of $NS: model=$M adapter=$AD"
pids=()
for s in "$@"; do
  /shared/egoavu/env/bin/python /ai4good1-shared/liyu/tools/h/nat_infer.py $K/requests/$BENCH.jsonl $O $M $AD $s $NS >> $O/worker_$s.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for s in "$@"; do tail -1 $O/worker_$s.log; done
h_log "nat $BENCH $MODEL shards $* of $NS exit $rc"; exit $rc
