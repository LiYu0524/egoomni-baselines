#!/usr/bin/env bash
# Submit one H rjob following the team's 8gpu.sh pattern.
# NOT --privileged: a privileged pod sees every GPU of the node; brainpp.cn/fuse=1 alone gives /dev/fuse + mount capability.
# usage: submit_h.sh NAME ai4good1|safevlagent GPUS [--dry-run] -- cmd args...
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
NAME=$1; PART=$2; GPU=$3; shift 3; DRY=false
[ "${1:-}" = "--dry-run" ] && { DRY=true; shift; }
[ "${1:-}" = "--" ] && shift
case $PART in
  ai4good1)    NS=ailab-ai4good1;    CG=ai4good1_gpu ;;
  safevlagent) NS=ailab-safevlagent; CG=safevlagent_gpu ;;
  *) echo "partition must be ai4good1|safevlagent"; exit 2 ;;
esac
CPU=$(( GPU * 8 )); MEM=$(( GPU * 64000 )); [ "$GPU" -eq 0 ] && { CPU=8; MEM=32000; }   # template: 4 GPU -> 32 CPU / 256 GB
CPU=${H_CPU:-$CPU}; MEM=${H_MEM:-$MEM}   # overrides for CPU-only jobs
rjob submit --name=$NAME --namespace=$NS --charged-group=$CG --private-machine=group \
  --gpu=$GPU --cpu=$CPU --memory=$MEM --replicas=1 \
  --mount=gpfs://gpfs1/ai4good1-share:/mnt/shared-storage-user/ai4good1-share \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --mount=gpfs://gpfs1/liyu:/mnt/shared-storage-user/liyu \
  --custom-resources brainpp.cn/fuse=1 \
  --image=${H_IMAGE:-registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab} \
  --dry-run=$DRY -- "$@"
