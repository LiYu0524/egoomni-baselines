#!/usr/bin/env bash
# Full pipeline, keeps all 8 GPUs busy back-to-back: inference for the 3 models (gold + self), then judge + score.
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh; cd $EGO_ROOT/eval
for m in videollama2 salmonn7b salmonn72b; do ./launch_infer.sh $m "$@"; done
CUDA_VISIBLE_DEVICES=0,1 ./launch_judge.sh videollama2_7b_av salmonn2plus_7b salmonn2plus_72b
for t in videollama2_7b_av salmonn2plus_7b salmonn2plus_72b; do for p in gold self; do $EGO_ENVS/judge/bin/python -m egoomni_eval.score --tag $t --protocol $p | tail -1; done; done
echo "[$(date -Is)] ALL DONE"
