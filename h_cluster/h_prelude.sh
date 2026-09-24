# h_prelude.sh — source at the top of every EgoAVU / egoOmni rjob on the H cluster.
#  1) P absolute paths (/ai4good1-shared/...) resolve to the H gpfs volume
#  2) s3://safevlagent/liyu/egoavu/ is mounted READ-ONLY at /mnt/egoavu_s3 (geesefs); dataset media is never copied to disk
#  3) h_restore_egoavu_env: EgoAVU runtime env from the PVC tarball onto the pod's local disk (/shared/egoavu)
H=/mnt/shared-storage-user/ai4good1-share; LY=$H/liyu
ln -sfn $H /ai4good1-shared
unset PYTHONPATH PYTHONHOME   # the CUDA image may ship its own LLaMA-Factory; only our patched venv may be imported
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY          # private GPU nodes have no internet; object storage is internal
export AWS_SHARED_CREDENTIALS_FILE=$LY/.aws/credentials AWS_CONFIG_FILE=$LY/.aws/config

h_log() { echo "[h $(date -Is) $(hostname)] $*"; }

h_mount_s3() {   # idempotent; fails loudly if the bucket is not readable
  local mnt=/mnt/egoavu_s3
  mountpoint -q $mnt && return 0
  mkdir -p $mnt /tmp/geesefs_cache
  $LY/tools/geesefs --endpoint http://hdd2.h.pjlab.org.cn:8060 --region us-east-1 \
      --shared-config $AWS_SHARED_CREDENTIALS_FILE --profile h-hdd2 \
      --memory-limit 8000 --stat-cache-ttl 1h --http-timeout 120s \
      --cache /tmp/geesefs_cache -o ro -o allow_other \
      safevlagent:liyu/egoavu $mnt || { h_log "geesefs mount FAILED"; return 1; }
  for i in $(seq 1 30); do mountpoint -q $mnt && break; sleep 1; done
  ls $mnt/proxy262k >/dev/null 2>&1 || { h_log "bucket mounted but proxy262k/ not visible"; return 1; }
  h_log "s3 media mounted at $mnt ($(ls $mnt | tr '\n' ' '))"
}

h_mount_egoomni() {   # s3://safevlagent/liyu/egoOmni/ (HF grooLegend/egoOmni mirror + ego4d_full sources) read-only
  local mnt=/mnt/egoomni_s3
  mountpoint -q $mnt && return 0
  mkdir -p $mnt /tmp/geesefs_cache_egoomni
  $LY/tools/geesefs --endpoint http://hdd2.h.pjlab.org.cn:8060 --region us-east-1 \
      --shared-config $AWS_SHARED_CREDENTIALS_FILE --profile h-hdd2 \
      --memory-limit 8000 --stat-cache-ttl 1h --http-timeout 120s \
      --cache /tmp/geesefs_cache_egoomni -o ro -o allow_other \
      safevlagent:liyu/egoOmni $mnt || { h_log "geesefs egoOmni mount FAILED"; return 1; }
  for i in $(seq 1 30); do mountpoint -q $mnt && break; sleep 1; done
  ls $mnt/test/clips >/dev/null 2>&1 || { h_log "egoOmni mounted but test/clips/ not visible"; return 1; }
  h_log "egoOmni media mounted at $mnt"
}

h_check_gpus() {   # usage: h_check_gpus N — refuse to run if the pod sees more/fewer GPUs than it was allocated
  local n; n=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU ")
  [ "$n" -eq "$1" ] || { h_log "GPU isolation check FAILED: pod sees $n GPUs, expected $1"; return 1; }
  h_log "GPU check ok: $(nvidia-smi -L | cut -c1-60 | tr "\n" ";")"
}

h_cuda_home() {   # LLaMA-Factory jobs: `import deepspeed` runs $CUDA_HOME/bin/nvcc -V; ml-base has no CUDA toolkit -> CUDA 12.6 nvcc on gpfs
  [ -x /usr/local/cuda/bin/nvcc ] && return 0
  [ -x $LY/tools/cuda126/bin/nvcc ] || { h_log "no nvcc available (tools/cuda126 missing)"; return 1; }
  export CUDA_HOME=$LY/tools/cuda126 CUDA_PATH=$LY/tools/cuda126
  h_log "CUDA_HOME=$CUDA_HOME ($($CUDA_HOME/bin/nvcc -V | tail -1))"
}

h_mount_prefix() {   # h_mount_prefix BUCKET_PREFIX MOUNTPOINT — read-only geesefs mount of s3://safevlagent/<prefix>
  local pre=$1 mnt=$2 c=/tmp/geesefs_cache_$(basename $2)
  mountpoint -q $mnt && return 0
  mkdir -p $mnt $c
  $LY/tools/geesefs --endpoint http://hdd2.h.pjlab.org.cn:8060 --region us-east-1 \
      --shared-config $AWS_SHARED_CREDENTIALS_FILE --profile h-hdd2 \
      --memory-limit 8000 --stat-cache-ttl 1h --http-timeout 120s --cache $c -o ro -o allow_other \
      safevlagent:$pre $mnt || { h_log "geesefs mount of $pre FAILED"; return 1; }
  for i in $(seq 1 30); do mountpoint -q $mnt && break; sleep 1; done
  mountpoint -q $mnt && h_log "s3://safevlagent/$pre mounted at $mnt"
}
