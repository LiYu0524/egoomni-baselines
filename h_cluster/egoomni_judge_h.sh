#!/usr/bin/env bash
# H rjob (1 GPU): egoOmni judge (local Qwen3-32B from gpfs2-shared-public, vLLM TP1) for the given tags, then score gold + self
# on (a) every evaluated item, as on P, and (b) the final bench only. A failed judge run stops the job (nothing is scored).
# usage: egoomni_judge_h.sh TAG [TAG ...]
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee -a $LY/tools/h/logs/egoomni_judge_$(hostname).log) 2>&1
h_check_gpus 1 || exit 1
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
export EGO_ENVS=$LY/egoOmni_baselines/envs_h; export PATH=$EGO_ENVS/judge/bin:$PATH   # vLLM compile step needs ninja
PY=$EGO_ENVS/judge/bin/python; cd $EGO_ROOT/eval; mkdir -p logs
J32=$($PY $LY/tools/h/pick_snapshot.py /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-32B) || exit 1
h_log "judge model: $J32 | $($PY -c 'import vllm, transformers; print("vllm", vllm.__version__, "transformers", transformers.__version__)')"
for TAG in "$@"; do
  $PY -m egoomni_eval.judge --tag $TAG --backend vllm --judge_model $J32 --tp 1 --gpu_util 0.85 >> logs/judge_${TAG}.log 2>&1
  rc=$?; grep -E "^\[judge\]" logs/judge_${TAG}.log | tail -2
  [ $rc -eq 0 ] || { h_log "judge $TAG FAILED (rc=$rc, see eval/logs/judge_${TAG}.log) - not scoring"; exit 1; }
  python3 $LY/tools/h/filter_final_preds.py $TAG
  for p in gold self; do
    $PY -m egoomni_eval.score --tag $TAG --protocol $p | head -14
    $PY -m egoomni_eval.score --tag $TAG --protocol $p --pred_root $EGO_ROOT/eval/preds_final --out_root $EGO_ROOT/eval/results_final | head -14
  done
  h_log "egoOmni $TAG scored"
done
h_log "egoOmni judge DONE"
