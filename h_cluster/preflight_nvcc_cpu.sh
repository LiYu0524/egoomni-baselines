#!/usr/bin/env bash
# CPU check of the exact call that crashed LLaMA-Factory: deepspeed op_builder.installed_cuda_version() via $CUDA_HOME/bin/nvcc.
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee $LY/tools/h/logs/preflight_nvcc_cpu_$(hostname).log) 2>&1
h_cuda_home || exit 1
tar -xf $LY/egoavu/backup/egoavu_nvme_20260921.tar -C /
/shared/egoavu/env/bin/python - <<'PY'
import torch.utils.cpp_extension as ce
print("[pfn] torch CUDA_HOME =", ce.CUDA_HOME)
from deepspeed.ops.op_builder.builder import installed_cuda_version
print("[pfn] installed_cuda_version ->", installed_cuda_version())
from deepspeed.ops.op_builder.fp_quantizer import FPQuantizerBuilder
print("[pfn] FPQuantizerBuilder.is_compatible ->", FPQuantizerBuilder().is_compatible(verbose=False))
PY
h_log "preflight nvcc cpu done"
