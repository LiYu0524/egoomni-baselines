#!/usr/bin/env bash
# H rjob: EgoAVU-Bench inference for one LoRA on one shard (media read from object storage, never copied).
# usage: bench_h.sh TAG ADAPTER_DIR|none DATASET [NPROC]
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
h_mount_s3 || exit 1
exec bash /ai4good1-shared/liyu/egoavu/bench_infer/tools/run_bench_infer.sh "$@"
