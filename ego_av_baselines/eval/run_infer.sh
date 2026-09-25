#!/bin/bash
# rjob entry for the EgoSound / EgoTempo / EgoToM baseline evals (1 GPU, no internet).
# usage: run_infer.sh MODEL BENCH NUM_SHARDS FIRST_SHARD PROCS
#   MODEL in qwen2vl | llavaov | minicpmo | egogpt | salmonn2p ; runs shards FIRST_SHARD .. FIRST_SHARD+PROCS-1 of NUM_SHARDS
#   concurrently on the pod's single GPU, after staging BENCH's media to local disk.
set -u
LY=/mnt/shared-storage-user/ai4good1-share/liyu; E=$LY/egobench_eval; M=$LY/models_bench
source $LY/tools/h/h_prelude.sh
MODEL=$1; BENCH=$2; NSH=$3; FIRST=$4; PROCS=$5
# media on the pod's overlay disk: /tmp is memory-backed (staging EgoSound there got the job OOM-killed)
export MEDIA_ROOT=/egobench_media HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
h_check_gpus 1 || exit 1
h_log "disks: $(df -h / /tmp | tail -n 2 | awk '{print $6"="$1"("$4" free)"}' | tr '\n' ' ')"
$LY/tools/egoclip_env/bin/python $E/stage_media.py $MEDIA_ROOT $BENCH || { h_log "staging failed"; exit 1; }
case $MODEL in
  qwen2vl)   PY=$LY/egoOmni_baselines/envs_h/judge/bin/python; export PYTHONPATH=$LY/tools/nat/site
             CMD="$E/infer_hf.py qwen2vl $M/Qwen2-VL-7B-Instruct" ;;
  llavaov)   PY=$LY/egoOmni_baselines/envs_h/judge/bin/python; export PYTHONPATH=$LY/tools/nat/site
             CMD="$E/infer_hf.py llavaov $M/llava-onevision-qwen2-7b-ov-hf" ;;
  minicpmo)  PY=$LY/envs_bench/minicpmo/bin/python; CMD="$E/infer_minicpmo.py $M/MiniCPM-o-2_6" ;;
  egogpt)    PY=$LY/envs_bench/egogpt/bin/python;   export EGOGPT_CWD=$M/egogpt_cwd
             CMD="$E/infer_egogpt.py $M/EgoGPT-7b-EgoIT-EgoLife" ;;
  salmonn2p) PY=$LY/envs_bench/salmonn2p/bin/python   # torchcodec 0.4 needs FFmpeg 4-7 shared libs (envs_h/ffmpeg is FFmpeg 8)
             export LD_LIBRARY_PATH=$LY/envs_bench/ffmpeg7/lib:${LD_LIBRARY_PATH:-}
             CMD="$E/infer_salmonn2p.py $M/video-SALMONN2_plus_7B_full" ;;
  *) h_log "unknown model $MODEL"; exit 2 ;;
esac
OUT=$E/preds/$MODEL/$BENCH; mkdir -p $OUT $E/logs
h_log "model=$MODEL bench=$BENCH shards $FIRST..$((FIRST + PROCS - 1)) of $NSH"
for s in $(seq $FIRST $((FIRST + PROCS - 1))); do
  $PY $CMD $E/requests/$BENCH.jsonl $OUT $s $NSH > $E/logs/${MODEL}_${BENCH}_s${s}.log 2>&1 &
done
wait
h_log "all shards finished"
