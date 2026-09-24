#!/usr/bin/env bash
# Build the egoOmni judge env (vLLM for the local Qwen3-32B grader) on gpfs, from openvla (H GPU nodes have no internet).
# Same recipe as scripts/build_env_judge.sh on P: python 3.10 + vllm + openai + ninja.
set -euo pipefail
L=/mnt/shared-storage-user/ai4good1-share/liyu
E=$L/egoOmni_baselines/envs_h/judge
export CONDA_PKGS_DIRS=$L/.cache/conda_pkgs PIP_CACHE_DIR=$L/.cache/pip   # openvla root disk has ~5 GB free
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR $(dirname $E)
echo "[judge-env] $(date -Is) start"
[ -x $E/bin/python ] || /root/miniconda3/bin/conda create -y -q -p $E python=3.10
$E/bin/python -m pip install -q --index-url https://pypi.org/simple --upgrade pip
$E/bin/python -m pip install -q --index-url https://pypi.org/simple "vllm==0.10.1.1" openai ninja   # cu12 build; H mirror 504s on big wheels
$E/bin/python -c "import vllm, torch; print('[judge-env] vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
du -sh $E
echo "[judge-env] $(date -Is) done"
