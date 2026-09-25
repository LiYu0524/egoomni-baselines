#!/bin/bash
# Build the per-model inference envs on gpfs as venvs layered on the judge env (torch/torchaudio/torchvision 2.7.1+cu128,
# triton 3.3.1): the internal pypi mirror times out (504) on multi-GB torch wheels, so only the model-specific pins are
# installed into each venv (they shadow the judge env's versions). Run sequentially (shared pip cache).
# usage: build_envs.sh NAME [NAME ...]      NAME in minicpmo | egogpt | salmonn2p
set -u
LY=/mnt/shared-storage-user/ai4good1-share/liyu; R=$LY/repos_bench; V=$LY/envs_bench; mkdir -p $V
J=$LY/egoOmni_baselines/envs_h/judge
PX=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
for name in "$@"; do
  P=$V/$name
  [ -x $P/bin/python ] || $J/bin/python -m venv --system-site-packages $P || exit 1
  PIP="$P/bin/pip install -q --no-cache-dir --retries 10 --timeout 120"
  # PyAV: the internal mirror 504s on the newest wheel -> older wheel from the mirror, else pypi.org via the proxy
  $PIP "av==12.3.0" || env https_proxy=$PX http_proxy=$PX $PIP --index-url https://pypi.org/simple "av==12.3.0"
  case $name in
    minicpmo)   # MiniCPM-o 2.6 model-card pins (torch from the judge env instead of 2.3.1)
      $PIP numpy==1.26.4 Pillow==10.1.0 transformers==4.44.2 "tokenizers>=0.19,<0.20" librosa==0.9.0 soundfile==0.12.1 \
           vector-quantize-pytorch==1.18.5 vocos==0.1.0 decord "moviepy==1.0.3" accelerate timm sentencepiece tqdm ;;
    egogpt)     # EgoLife/EgoGPT pyproject inference pins (torch from the judge env instead of 2.1.2) + the package
      $PIP numpy==1.26.4 transformers==4.45.2 tokenizers==0.20.1 sentencepiece accelerate==1.0.1 peft==0.11.1 \
           einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 decord soundfile scipy openai-whisper shortuuid tqdm \
        && $P/bin/pip install -q --no-deps -e $R/EgoLife/EgoGPT ;;
    salmonn2p)  # video_SALMONN2_plus/requirements.txt (torch 2.7.1 matches); flash-attn prebuilt wheel via the proxy
      $PIP numpy==1.24.4 transformers==4.51.3 tokenizers==0.21.0 accelerate==1.7.0 peft==0.15.2 decord==0.6.0 \
           torchcodec==0.4.0 liger_kernel==0.5.10 soundfile tqdm \
        && { env https_proxy=$PX http_proxy=$PX $PIP --no-deps \
               "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" \
             || echo "flash-attn wheel unavailable - adapter falls back to sdpa"; } ;;
    *) echo "unknown env $name"; continue ;;
  esac
  echo "BUILD_${name}_RC=$?"
  $P/bin/python -c "import torch, transformers, numpy; print('$name', torch.__version__, transformers.__version__, numpy.__version__)"
done
