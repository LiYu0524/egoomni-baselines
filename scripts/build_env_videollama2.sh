#!/usr/bin/env bash
# VideoLLaMA2 (audio_visual branch) env: torch 2.2.0 / transformers 4.42.3 / flash-attn 2.5.8  (repo requirements.txt)
set -euo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
E=$EGO_ENVS/videollama2; REPO=$EGO_ROOT/repos/VideoLLaMA2
[ -x $E/bin/python ] || conda create -y -q -p $E python=3.10
conda install -y -q -p $E -c conda-forge ffmpeg    # for moviepy/torchaudio audio extraction
PIP="$E/bin/python -m pip"
$PIP install -q --upgrade pip
$PIP install -q -r $REPO/requirements.txt           # pulls torch 2.2.0+cu118 via the extra-index in the file
( source $EGO_ROOT/.proxy_env; $PIP install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" )
$PIP install -q opencv-python==4.5.5.64             # README step
$PIP install -q "setuptools<81"                     # imageio-ffmpeg 0.4.9 needs pkg_resources
$PIP install -q -e $REPO --no-deps
$E/bin/python - <<PY
import torch, transformers, flash_attn, decord, librosa, videollama2
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| transformers", transformers.__version__, "| flash_attn", flash_attn.__version__)
PY
echo "[$(date -Is)] videollama2 env OK"
