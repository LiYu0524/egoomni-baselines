#!/usr/bin/env bash
# Resume pass for the slow tail of the Gemini restored_v3 run: remaining items spread over 12 workers → restored_v3/resume/
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh; source $EGO_ROOT/.gemini_env; cd $EGO_ROOT/eval
export FFMPEG=$EGO_ENVS/minicpmo/bin/ffmpeg EGO_QA_PATH=/ai4good1-shared/liyu/egoOmni/test/qa_restored_v3.json EGO_CLIPS_META=$EGO_ROOT/eval/clips_meta_restored_v3.json
for i in $(seq 0 11); do
  python3 -m egoomni_eval.run_infer --adapter gemini --model_dir gemini-3.8-flash --tag gemini_3_8_flash --out_name gemini_3_8_flash/restored_v3/resume \
    --subset subsets/restored_v3_gemini_remaining.json --shard $i --nshards 12 --prefetch 1 > logs/infer_gemini_restored_v3_resume_shard$i.log 2>&1 &
done
wait; echo "[$(date -Is)] resume finished"
