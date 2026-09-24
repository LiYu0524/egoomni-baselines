#!/usr/bin/env bash
# H rjob (CPU only): 2 fps proxies of the egoOmni test clips for the EgoToM-kit re-test (tools/h/prep_egoomni_proxy2fps.py).
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee -a $LY/tools/h/logs/prep_egoomni_proxy2fps_$(hostname).log) 2>&1
h_mount_egoomni || exit 1
h_log "start ($(nproc) CPUs)"
$LY/egoOmni_baselines/envs_h/judge/bin/python /ai4good1-shared/liyu/tools/h/prep_egoomni_proxy2fps.py ${1:-10} && h_log "proxy prep done" || { h_log "proxy prep FAILED"; exit 1; }
