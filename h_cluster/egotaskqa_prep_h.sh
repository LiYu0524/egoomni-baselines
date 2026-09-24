#!/usr/bin/env bash
# CPU rjob: EgoTaskQA test-clip proxies (2 fps / ~262k px / 16 kHz FLAC), streamed out of qa_videos.zip in object storage.
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee -a $LY/egotaskqa_eval_h/logs/prep_$(hostname).log) 2>&1
h_mount_prefix liyu/benchmarks/egotaskqa/data /mnt/egotaskqa_s3 || exit 1
FFENV=$LY/egoOmni_baselines/envs_h/ffmpeg; mkdir -p /shared/egoOmni_envs/minicpmo/bin
ln -sf $FFENV/bin/ffmpeg /shared/egoOmni_envs/minicpmo/bin/ffmpeg; ln -sf $FFENV/bin/ffprobe /shared/egoOmni_envs/minicpmo/bin/ffprobe
$LY/egoOmni_baselines/envs_h/judge/bin/python /ai4good1-shared/liyu/egotaskqa_eval_h/prep_media.py ${1:-14} || { h_log "egotaskqa prep FAILED"; exit 1; }
h_log "egotaskqa prep done"
