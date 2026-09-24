#!/usr/bin/env bash
# H rjob (1 H200): EgoAVU-Bench inference with TWO bs=1 predict processes sharing the card (8-way shards A and B).
# Each process is a normal 1-GPU run, so per-sample outputs equal a single-process run; the 2nd one just fills the card
# (7B inference peaks ~30 GB of 141 GB; bs=1 decode alone leaves the GPU mostly idle).
# usage: bench2_h.sh TAG ADAPTER_DIR SHARD_A SHARD_B
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
h_check_gpus 1 && h_cuda_home && h_mount_s3 || exit 1
PVC=/ai4good1-shared/liyu/egoavu
[ -x /shared/egoavu/env/bin/python ] || tar -xf $PVC/backup/egoavu_nvme_20260921.tar -C /   # once, before the parallel runs
TAG=$1; AD=$2; shift 2; pids=(); port=29640
for s in "$@"; do
  [ -f $PVC/bench_infer/runs/$TAG/egoavu_bench_eval_s${s}of8/collect_stats.json ] && { h_log "shard $s already collected - skip"; continue; }
  MASTER_PORT=$port OMP_NUM_THREADS=6 bash $PVC/bench_infer/tools/run_bench_infer.sh $TAG $AD egoavu_bench_eval_s${s}of8 1 &
  pids+=($!); port=$((port + 10))
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
h_log "bench2 $TAG shards $* exit $rc"; exit $rc
