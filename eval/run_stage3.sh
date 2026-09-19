#!/usr/bin/env bash
# Stage 3: judge + score as runs complete; then the preprocessing-deviation study (repo CPU path vs our GPU path on dev300).
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh; cd $EGO_ROOT/eval; mkdir -p logs results
log() { echo "[$(date -Is)] $*"; }
wait_pid() { while kill -0 $1 2>/dev/null; do sleep 60; done; }
J=$EGO_ENVS/judge/bin/python
score_all() { for p in gold self; do $J -m egoomni_eval.score --tag $1 --protocol $p | tail -1; done; }
# 1) videollama2: judge already running → wait for it, score
while pgrep -f "^/shared/egoOmni_envs/judge/bin/python -m egoomni_eval.judge" >/dev/null; do sleep 30; done
score_all videollama2_7b_av
# 2) salmonn 7b: wait for its launcher, judge co-resident with whatever runs (TP=8 @ 20%), score
[ -n "${SALMONN7B_PID:-}" ] && { log "waiting for salmonn7b launcher $SALMONN7B_PID"; wait_pid $SALMONN7B_PID; }
sleep 120   # stage2's error-retry pass, if any, is quick
GPU_UTIL=0.2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./launch_judge.sh salmonn2plus_7b
score_all salmonn2plus_7b
# 3) 72b: wait for stage 2, judge, score
[ -n "${STAGE2_PID:-}" ] && { log "waiting for stage2 $STAGE2_PID"; wait_pid $STAGE2_PID; }
if [ -f STAGE2_ABORT ]; then log "stage 2 aborted — skipping 72B judge"; else
  GPU_UTIL=0.85 CUDA_VISIBLE_DEVICES=0,1 ./launch_judge.sh salmonn2plus_72b
  score_all salmonn2plus_72b
fi
# 4) deviation study: repo CPU preprocessing (exact) vs GPU preprocessing on dev300, SALMONN 7B, gold protocol only
log "deviation study: salmonn7b dev300 with --no_gpu_preprocess (repo-exact CPU path)"
NGPU=8 ./launch_infer.sh salmonn7b --subset subsets/dev300.json --protocols gold --no_gpu_preprocess --cpu_threads 16 --out_name salmonn2plus_7b_cpuprep_dev300
GPU_UTIL=0.85 CUDA_VISIBLE_DEVICES=0,1 ./launch_judge.sh salmonn2plus_7b_cpuprep_dev300
$J -m egoomni_eval.score --tag salmonn2plus_7b_cpuprep_dev300 --protocol gold | tail -1
$J - <<'PY'
import json, glob
from egoomni_eval.judge import pred_hash
def rows(tag): return {r["key"]: r for f in glob.glob(f"preds/{tag}/shard*.jsonl") for l in open(f) if l.strip() for r in [json.loads(l)] if r["protocol"]=="gold"}
def judg(tag): return {(j["key"], j["pred_hash"]): j["correct"] for j in (json.loads(l) for l in open(f"judgments/{tag}.jsonl")) if j["correct"] is not None}
cpu, gpu = rows("salmonn2plus_7b_cpuprep_dev300"), rows("salmonn2plus_7b"); jc, jg = judg("salmonn2plus_7b_cpuprep_dev300"), judg("salmonn2plus_7b")
keys = [k for k in cpu if k in gpu and cpu[k]["pred"] is not None and gpu[k]["pred"] is not None]
same = sum(cpu[k]["pred"].strip() == gpu[k]["pred"].strip() for k in keys)
open_keys = [k for k in keys if cpu[k]["fmt"] != "mcq"]
acc = lambda R, J: sum(bool(J.get((k, pred_hash(R[k]["pred"])))) for k in open_keys) / max(len(open_keys), 1)
res = {"n_compared": len(keys), "identical_predictions": same, "identical_frac": round(same/len(keys), 4), "open_acc_cpu_path": round(acc(cpu, jc), 4), "open_acc_gpu_path": round(acc(gpu, jg), 4),
       "verdict_agreement": round(sum(bool(jc.get((k, pred_hash(cpu[k]["pred"])))) == bool(jg.get((k, pred_hash(gpu[k]["pred"])))) for k in open_keys)/max(len(open_keys),1), 4)}
json.dump(res, open("results/preprocessing_deviation_dev300.json", "w"), indent=2); print(res)
PY
log "STAGE 3 DONE"
