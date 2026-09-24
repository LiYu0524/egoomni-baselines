#!/usr/bin/env bash
# 1-GPU preflight of the CUDA image: GPU isolation, nvcc, env vars, and the exact import that failed (deepspeed op-builder check).
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee $LY/tools/h/logs/preflight_cuda_img_$(hostname).log) 2>&1
h_check_gpus 1 && h_cuda_home || exit 1
h_log "image env: PYTHONPATH=${PYTHONPATH:-} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} CUDA_HOME=${CUDA_HOME:-} CUDA_PATH=${CUDA_PATH:-}"
h_log "nvcc: $(/usr/local/cuda/bin/nvcc -V 2>&1 | tail -1)"; grep PRETTY /etc/os-release
t=$(date +%s); tar -xf $LY/egoavu/backup/egoavu_nvme_20260921.tar -C / && h_log "env restored in $(( $(date +%s) - t )) s"
/shared/egoavu/env/bin/python - <<'PY'
import torch, llamafactory, deepspeed, transformers
from deepspeed.ops.op_builder.builder import installed_cuda_version
from llamafactory.data.mm_plugin import _egoavu_split_fragment
print("[pfc] llamafactory from", llamafactory.__file__)
print("[pfc] torch", torch.__version__, "cuda ok", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("[pfc] deepspeed", deepspeed.__version__, "installed_cuda_version", installed_cuda_version())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); print("[pfc] matmul ok", float((x @ x).float().abs().mean()))
PY
h_log "preflight cuda image done"
