#!/usr/bin/env bash
# How the ckpt_sft run was submitted to the P-cluster queue (clusterx CLI). Volume id / registry are placeholders.
# The ckpt_epoch2 run used the same body on an 8-GPU dev box:  MASTER_PORT=29650 bash tools/run_bench_infer.sh ckpt_epoch2 <adapter_dir> egoavu_bench_eval 8
BI=/ai4good1-shared/liyu/egoavu/bench_infer
clusterx run -J egoavu-bench-sft -N 1 --gpus-per-task 8 --cpus-per-task 48 --memory-per-task 400 --shm-size-gib 64 --no-env \
  --image <REGISTRY>/ngc-pytorch:25.06-cu12.9-py3.12-ubuntu24.04 \
  --mount PV_AFS:<AFS_VOLUME_ID>:/ai4good1-shared \
  bash $BI/tools/run_bench_infer.sh ckpt_sft $BI/adapters/ckpt_sft egoavu_bench_eval 8
