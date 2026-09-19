# shared env for egoOmni_baselines scripts:  source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
export EGO_ROOT=/ai4good1-shared/liyu/egoOmni_baselines
export PATH=$EGO_ROOT/miniconda3/bin:$PATH
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_EXTRA_INDEX_URL=https://pypi.org/simple
export PIP_TIMEOUT=120
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120
export MODELSCOPE_CACHE=$EGO_ROOT/models/.modelscope_cache
export EGO_ENVS=/shared/egoOmni_envs          # runtime envs live on local NVMe (PVC is ~85x slower on small files); tarball backups in $EGO_ROOT/envs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # everything is local; never hit the Hub at runtime
