#!/usr/bin/env bash
# Build the small conda envs the H jobs need, bypassing openvla's internal conda-forge mirror (it 404s on many packages).
# Order: CUDA 12.6 nvcc (DeepSpeed import needs `nvcc -V`) -> ffmpeg (egoOmni media prep) -> capeval (METEOR needs Java).
L=/mnt/shared-storage-user/ai4good1-share/liyu
export CONDA_PKGS_DIRS=$L/.cache/conda_pkgs PIP_CACHE_DIR=$L/.cache/pip
export https_proxy=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 http_proxy=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128
CF="--override-channels -c https://conda.anaconda.org/conda-forge"
mk() { local p=$1; shift; [ -e $p/conda-meta/history ] && [ -x "$p/bin/$MUST" ] && { echo "[envs] $p exists"; return 0; }
       rm -rf $p; /root/miniconda3/bin/conda create -y -q -p $p $CF "$@" || { echo "[envs] FAILED $p"; return 1; }; }
echo "[envs] $(date -Is) start"
MUST=nvcc  mk $L/tools/cuda126 "cuda-nvcc=12.6" && $L/tools/cuda126/bin/nvcc -V | tail -1 && echo "[envs] $(date -Is) cuda126 ok"
MUST=ffmpeg mk $L/egoOmni_baselines/envs_h/ffmpeg ffmpeg && $L/egoOmni_baselines/envs_h/ffmpeg/bin/ffmpeg -hide_banner -version | head -1 && echo "[envs] $(date -Is) ffmpeg ok"
E=$L/egoOmni_baselines/envs_h/capeval
MUST=java mk $E python=3.10 openjdk=11 && $E/bin/python -m pip install -q --index-url https://pypi.org/simple pycocoevalcap tqdm \
  && $E/bin/python -c "from pycocoevalcap.meteor.meteor import Meteor; from pycocoevalcap.rouge.rouge import Rouge; from pycocoevalcap.cider.cider import Cider; print('[envs] capeval imports ok')" \
  && echo "[envs] $(date -Is) capeval ok"
echo "[envs] $(date -Is) done"
