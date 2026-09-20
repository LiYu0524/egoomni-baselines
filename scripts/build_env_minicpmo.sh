#!/usr/bin/env bash
# MiniCPM-o 2.6 env: pins from the model card (transformers 4.44.2, torch 2.3.1, librosa 0.9.0, vocos, vector-quantize-pytorch)
set -euo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
E=$EGO_ENVS/minicpmo
[ -x $E/bin/python ] || conda create -y -q -p $E python=3.10
conda install -y -q -p $E -c conda-forge "ffmpeg=7.*"    # audio extraction (ffmpeg subprocess) + decord
PIP="$E/bin/python -m pip"
$PIP install -q --upgrade pip "setuptools<81"
$PIP install -q torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
$PIP install -q "numpy<2" Pillow==10.1.0 transformers==4.44.2 librosa==0.9.0 soundfile==0.12.1 vector-quantize-pytorch==1.18.5 vocos==0.1.0 decord==0.6.0 moviepy==1.0.3 accelerate sentencepiece einops
$E/bin/python - <<PY
import torch, transformers, decord, librosa, vocos, vector_quantize_pytorch, PIL
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| transformers", transformers.__version__, "| PIL", PIL.__version__, "| librosa", librosa.__version__)
PY
echo "[$(date -Is)] minicpmo env OK"
