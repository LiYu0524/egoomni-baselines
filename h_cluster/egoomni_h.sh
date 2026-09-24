#!/usr/bin/env bash
# H rjob: egoOmni bench (gold, then self rounds 2..4) for one EgoAVU LoRA instance, shard k of K. Media via the s3 mount.
# usage: egoomni_h.sh INSTANCE(r100k|r20k8g|r20k32g) K k [NPROC]
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
h_mount_egoomni || exit 1
exec bash /ai4good1-shared/liyu/egoOmni_baselines/eval/$1/job.sh "${@:2}"
