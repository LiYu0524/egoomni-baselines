#!/usr/bin/env bash
# Env for the official EgoAVU captioning_eval.py (pycocoevalcap METEOR/ROUGE/CIDEr; METEOR needs Java). Built on gpfs from openvla.
set -euo pipefail
L=/mnt/shared-storage-user/ai4good1-share/liyu; E=$L/egoOmni_baselines/envs_h/capeval
export CONDA_PKGS_DIRS=$L/.cache/conda_pkgs PIP_CACHE_DIR=$L/.cache/pip
echo "[capeval] $(date -Is) start"
[ -x $E/bin/python ] || /root/miniconda3/bin/conda create -y -q -p $E -c conda-forge python=3.10 openjdk=11
$E/bin/python -m pip install -q --index-url https://pypi.org/simple pycocoevalcap tqdm
PATH=$E/bin:$PATH JAVA_HOME=$E $E/bin/java -version 2>&1 | head -1
$E/bin/python -c "from pycocoevalcap.meteor.meteor import Meteor; from pycocoevalcap.rouge.rouge import Rouge; from pycocoevalcap.cider.cider import Cider; print('[capeval] imports ok')"
du -sh $E; echo "[capeval] $(date -Is) done"
