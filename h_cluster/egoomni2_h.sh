#!/usr/bin/env bash
# H rjob (1 H200): egoOmni bench for one EgoAVU LoRA instance, TWO shards of K run concurrently on the card (1 process each).
# usage: egoomni2_h.sh INSTANCE K SHARD_A SHARD_B
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
h_check_gpus 1 && h_cuda_home && h_mount_egoomni || exit 1
[ -x /shared/egoavu/env/bin/python ] || tar -xf /ai4good1-shared/liyu/egoavu/backup/egoavu_nvme_20260921.tar -C /
INST=$1; K=$2; shift 2; pids=(); port=29641
for k in "$@"; do
  MASTER_PORT=$port bash /ai4good1-shared/liyu/egoOmni_baselines/eval/$INST/job.sh $K $k 1 &
  pids+=($!); port=$((port + 10))
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
h_log "egoomni2 $INST shards $* of $K exit $rc"; exit $rc
