#!/usr/bin/env bash
# Stage 3b (replaces stage 3 after the co-resident 7B judge OOMed): wait for stage 2 (72B inference) → judge 7B and 72B
# concurrently on disjoint GPU halves → score all → preprocessing-deviation study → final summary.
set -uo pipefail
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh; cd $EGO_ROOT/eval; mkdir -p logs results
log() { echo "[$(date -Is)] $*"; }
wait_pid() { while kill -0 $1 2>/dev/null; do sleep 60; done; }
J=$EGO_ENVS/judge/bin/python
score_all() { for p in gold self; do $J -m egoomni_eval.score --tag $1 --protocol $p | tail -1; done; }
[ -n "${STAGE2_PID:-}" ] && { log "waiting for stage2 $STAGE2_PID (72B inference)"; wait_pid $STAGE2_PID; }
[ -f STAGE2_ABORT ] && log "WARNING: stage 2 aborted; 72B may be incomplete"
rm -f results/salmonn2plus_7b/*.json results/salmonn2plus_7b/*.md      # invalid (unjudged) tables from the failed attempt
log "judging salmonn2plus_7b (GPUs 0-3) and salmonn2plus_72b (GPUs 4-7) concurrently"
GPU_UTIL=0.85 CUDA_VISIBLE_DEVICES=0,1,2,3 ./launch_judge.sh salmonn2plus_7b  > logs/judge_stage3b_7b.log  2>&1 &
GPU_UTIL=0.85 CUDA_VISIBLE_DEVICES=4,5,6,7 ./launch_judge.sh salmonn2plus_72b > logs/judge_stage3b_72b.log 2>&1 &
wait
for t in videollama2_7b_av salmonn2plus_7b salmonn2plus_72b; do log "scoring $t"; score_all $t; done
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
ok = [k for k in keys if cpu[k]["fmt"] != "mcq"]
acc = lambda R, Jd: sum(bool(Jd.get((k, pred_hash(R[k]["pred"])))) for k in ok) / max(len(ok), 1)
res = {"n_compared": len(keys), "identical_predictions": same, "identical_frac": round(same/max(len(keys),1), 4),
       "open_acc_cpu_path": round(acc(cpu, jc), 4), "open_acc_gpu_path": round(acc(gpu, jg), 4),
       "verdict_agreement": round(sum(bool(jc.get((k, pred_hash(cpu[k]["pred"])))) == bool(jg.get((k, pred_hash(gpu[k]["pred"])))) for k in ok)/max(len(ok),1), 4)}
json.dump(res, open("results/preprocessing_deviation_dev300.json", "w"), indent=2); print(res)
PY
$J - <<'PY'
import json
rows=["| model | overall (gold) | single-turn | multi-turn round (gold) | multi-turn round (self) | all-rounds-correct (gold) | MCQ | open |", "|---|---|---|---|---|---|---|---|"]
for t in ["videollama2_7b_av","salmonn2plus_7b","salmonn2plus_72b"]:
    try: g=json.load(open(f"results/{t}/gold.json")); s=json.load(open(f"results/{t}/self.json"))
    except FileNotFoundError: rows.append(f"| {t} | (missing) |||||||"); continue
    b=g["breakdowns"]["format"]
    rows.append(f"| {t} | {g['overall_acc']:.4f} | {g['single_turn_acc']:.4f} | {g['multi_turn_round_acc']:.4f} | {s['multi_turn_round_acc']:.4f} | {g['multi_turn_all_rounds_correct']:.4f} | {b.get('mcq',{}).get('acc',0):.4f} | {b.get('open',{}).get('acc',0):.4f} |")
open("results/SUMMARY.md","w").write("# egoOmni baselines — summary (judge: Qwen3-32B)\n\n"+"\n".join(rows)+"\n"); print("\n".join(rows))
PY
log "STAGE 3b DONE"
