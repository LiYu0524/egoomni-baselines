#!/usr/bin/env bash
# CPU rjob: EgoSchema proxies (2 fps / ~262k px / 16 kHz FLAC) for one split, streamed out of the zips in object storage.
# usage: egoschema_prep_h.sh subset|mc WORKERS
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee -a $LY/egoschema_eval_h/logs/prep_$1_$(hostname).log) 2>&1
h_mount_prefix liyu/benchmarks/egoschema /mnt/egoschema_s3 || exit 1
$LY/egoOmni_baselines/envs_h/judge/bin/python /ai4good1-shared/liyu/egoschema_eval_h/prep_media.py $1 ${2:-24} || { h_log "egoschema prep FAILED"; exit 1; }
h_log "egoschema prep done"
