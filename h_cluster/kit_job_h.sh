#!/usr/bin/env bash
# H rjob (1 H200): EgoToM-kit-protocol re-test of one model on EgoAVU-Bench or egoOmni — one process per given shard on this card.
#   eab:     tools/h/nat_predict_lf.py on data/egoavu_bench_eval_s{k}of{K} -> runs/kit_<model>/..., 1024 new tokens (the bench's
#            budget, gold answers reach 483 tokens), <=360 frames; then the unchanged collect_predictions.py (join checks)
#   egoomni: eval/kit_<model>/job.sh K k (gold + self rounds, 256 new tokens, <=360 frames)
# usage: kit_job_h.sh MODEL eab|egoomni K SHARD [SHARD ...]      MODEL = base | ckpt_epoch2 | ckpt_fft_epoch2
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
MODEL=$1; BENCH=$2; K=$3; shift 3
mkdir -p $LY/natkit/logs; exec > >(tee -a $LY/natkit/logs/kitjob_${BENCH}_${MODEL}_$(hostname).log) 2>&1
h_check_gpus 1 || exit 1
case $BENCH in
  eab) h_mount_s3 || exit 1 ;;
  egoomni) h_mount_egoomni || exit 1 ;;
  *) h_log "unknown benchmark $BENCH"; exit 2 ;;
esac
[ -x /shared/egoavu/env/bin/python ] || tar -xf /ai4good1-shared/liyu/egoavu/backup/egoavu_nvme_20260921.tar -C /
case $MODEL in
  base)            M=/ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train; AD=none ;;
  ckpt_epoch2)     M=/ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train; AD=/ai4good1-shared/liyu/egoavu/bench_infer/adapters/ckpt_epoch2 ;;
  ckpt_fft_epoch2) M=/ai4good1-shared/liyu/egoavu/models/groo_ckpt_fft_epoch2; AD=none ;;
  *) h_log "unknown model $MODEL"; exit 2 ;;
esac
FFB=/ai4good1-shared/liyu/egoOmni_baselines/envs_h/ffmpeg/bin
export PATH="/shared/egoavu/env/bin:$FFB:$PATH" PYTHONPATH=/ai4good1-shared/liyu/tools/nat/site FFPROBE=$FFB/ffprobe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NAT_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export FORCE_QWENVL_VIDEO_READER=decord PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
BI=/ai4good1-shared/liyu/egoavu/bench_infer; TAG=kit_$MODEL; PY=/shared/egoavu/env/bin/python
[ $BENCH = egoomni ] && export NAT_VIDEO_PREFIX_MAP="/ai4good1-shared/liyu/egoOmni/test/=>/ai4good1-shared/liyu/egoOmni_baselines/eval/kit_proxy2fps/"   # 2 fps proxies (decord cannot read the source clips)
h_log "kit $BENCH $MODEL shards $* of $K: model=$M adapter=$AD"
pids=()
for s in "$@"; do
  if [ $BENCH = eab ]; then
    ds=egoavu_bench_eval_s${s}of${K}
    ( [ -f $BI/runs/$TAG/$ds/collect_stats.json ] && exit 0
      $PY /ai4good1-shared/liyu/tools/h/nat_predict_lf.py $BI/data/$ds.jsonl $BI/runs/$TAG/$ds $M $AD 1024 360 >> $BI/logs/${TAG}__$ds.log 2>&1 \
        || { h_log "$ds predict FAILED"; exit 1; }
      cd /tmp && $PY $BI/tools/collect_predictions.py $TAG $ds > $BI/logs/${TAG}__${ds}_collect.log 2>&1 \
        || { h_log "$ds collect FAILED"; exit 1; }
      h_log "$ds done" ) &
  else
    bash /ai4good1-shared/liyu/egoOmni_baselines/eval/$TAG/job.sh $K $s &
  fi
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
h_log "kit $BENCH $MODEL shards $* of $K exit $rc"; exit $rc
