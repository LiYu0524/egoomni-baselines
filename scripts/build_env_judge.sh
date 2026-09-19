#!/usr/bin/env bash
# judge env: vLLM for the local Qwen3-32B grader (TP=2)
set -euo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
E=$EGO_ENVS/judge
[ -x $E/bin/python ] || conda create -y -q -p $E python=3.10
PIP="$E/bin/python -m pip"
$PIP install -q --upgrade pip
$PIP install -q vllm openai ninja
$E/bin/python -c "import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"
echo "[$(date -Is)] judge env OK"
