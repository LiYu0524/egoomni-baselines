#!/usr/bin/env bash
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee $LY/tools/h/logs/probe_nopriv_$(hostname).log) 2>&1
h_log "privileged? CapEff=$(grep CapEff /proc/self/status | awk "{print \$2}") /dev/fuse: $(ls -la /dev/fuse 2>&1)"
h_log "nvidia devices: $(ls /dev/nvidia* 2>/dev/null | tr "\n" " ")"
h_log "GPU env: $(env | grep -E "^(NVIDIA_|CUDA_|.*GPU.*=)" | tr "\n" " ")"
h_mount_s3 && h_log "geesefs WITHOUT privileged: OK, $(ls /mnt/egoavu_s3/proxy262k/test | wc -l) test files visible" || h_log "geesefs WITHOUT privileged: FAILED"
h_log "probe done"
