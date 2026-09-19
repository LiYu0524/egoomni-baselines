#!/usr/bin/env bash
# video-SALMONN 2+ env: torch 2.7.1 / transformers 4.51.3 / peft 0.15.2 / liger 0.5.10 (repo requirements.txt)
# deviations: torchaudio 2.7.1 (repo pins 2.5.1, incompatible with torch 2.7.1); flash-attn 2.8.0.post2 prebuilt (2.7.4.post1 has no torch-2.7 wheel)
set -euo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
E=$EGO_ENVS/salmonn2plus
[ -x $E/bin/python ] || conda create -y -q -p $E python=3.10
conda install -y -q -p $E -c conda-forge "ffmpeg=7.*"    # torchcodec 0.4.0 supports FFmpeg 4-7 only (conda default is 8)
PIP="$E/bin/python -m pip"
$PIP install -q --upgrade pip
$PIP install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 torchcodec==0.4.0
( source $EGO_ROOT/.proxy_env; $PIP install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" )
$PIP install -q accelerate==1.7.0 decord==0.6.0 deepspeed==0.16.0 liger_kernel==0.5.10 numpy==1.24.4 peft==0.15.2 tokenizers==0.21.0 transformers==4.51.3 triton==3.3.1 tqdm wandb
$E/bin/python - <<PY
import torch, transformers, flash_attn, peft, liger_kernel, decord, torchcodec
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| transformers", transformers.__version__, "| flash_attn", flash_attn.__version__, "| peft", peft.__version__)
PY
echo "[$(date -Is)] salmonn2plus env OK"
