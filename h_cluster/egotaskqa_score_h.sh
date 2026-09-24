#!/usr/bin/env bash
# H rjob (1 GPU): EgoTaskQA scoring for one model — exact match + Qwen3-32B judge (vLLM TP1), per split and breakdown.
# usage: egotaskqa_score_h.sh TAG
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee -a $LY/egotaskqa_eval_h/logs/score_$1_$(hostname).log) 2>&1
h_check_gpus 1 || exit 1
JE=$LY/egoOmni_baselines/envs_h/judge; export PATH=$JE/bin:$PATH
J32=$($JE/bin/python $LY/tools/h/pick_snapshot.py /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-32B) || exit 1
$JE/bin/python /ai4good1-shared/liyu/egotaskqa_eval_h/score_judge.py $1 $J32 || { h_log "EgoTaskQA score $1 FAILED"; exit 1; }
h_log "EgoTaskQA $1 scored"
