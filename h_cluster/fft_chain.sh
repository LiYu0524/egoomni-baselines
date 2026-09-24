#!/usr/bin/env bash
# openvla: evaluate groo_legend/ckpts ckpt_fft_epoch2 (full fine-tune, re-headered into the thinker view) on the five ego benchmarks
# with exactly the protocols / decoding the other models got. Waits for MODEL_READY (conversion verified). Smoke first: EgoCross +
# EgoSchema; everything else is submitted only after one of them has loaded the checkpoint cleanly (no newly initialized or unused
# weights). Then EgoTaskQA (4 jobs + scoring), EgoAVU-Bench (4 jobs x 2 shards + official judge on 4 GPUs into
# release_h_fft / judge_h_fft / results_h_fft) and egoOmni (4 jobs x 2 shards + Qwen3-32B judge).
# Failed or stuck (pod-sandbox) jobs are resubmitted up to 3 times.  Log: tools/h/logs/fft_chain.log
L=/mnt/shared-storage-user/ai4good1-share/liyu; T=$L/tools/h; ST=$T/state_fft; mkdir -p $ST
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
TAG=ckpt_fft_epoch2; J=ckpt-fft-epoch2; M=/ai4good1-shared/liyu/egoavu/models/groo_ckpt_fft_epoch2; AD=model=$M
BI=$L/egoavu/bench_infer; E=$L/egoOmni_baselines/eval; X=$L/egocross_eval_h; S=$L/egoschema_eval_h; Q=$L/egotaskqa_eval_h
log() { echo "[fft $(date -Is)] $*"; }
st() { brainctl get rjobs $1 -n ailab-safevlagent --no-headers 2>/dev/null | awk '{print $2, $3}'; }
age_min() { echo "$1" | awk '{t=0; s=$0; while (match(s, /[0-9]+[dhms]/)) {v=substr(s, RSTART, RLENGTH); n=substr(v, 1, length(v)-1)+0; u=substr(v, length(v));
  t += (u=="d") ? n*1440 : (u=="h") ? n*60 : (u=="m") ? n : 0; s=substr(s, RSTART+RLENGTH)} print t}'; }
cur() { local a=$(cat $ST/$1.attempts 2>/dev/null || echo 0); [ $a -eq 0 ] && echo $1 || echo $1-r$a; }
submit() { local n=$1 g=$2 c=$3 m=$4; shift 4
  H_CPU=$c H_MEM=$m $T/submit_h.sh $n safevlagent $g -- "$@" 2>&1 | grep -q "created rjob_name" && log "submitted $n" || log "SUBMIT FAILED $n"; }
tend() {   # KEY GPUS CPU MEM DONE_CMD -- cmd...   (all helper-function loop variables are local: bash locals are dynamically scoped)
  local key=$1 g=$2 c=$3 m=$4 d=$5; shift 6
  eval "$d" && return 0
  local n s state age a; n=$(cur $key); s=$(st $n)
  [ -z "$s" ] && { submit $n $g $c $m "$@"; return 1; }
  state=${s%% *}; age=${s##* }
  if [ "$state" = Starting ] && [ "$(age_min $age)" -ge 20 ] && KUBEBRAIN_NAMESPACE=ailab-safevlagent timeout 60 rjob events $n 2>/dev/null | grep -q "Failed to create pod sandbox"; then
    log "$n stuck in Starting ($age, pod-sandbox errors) - stopping"; KUBEBRAIN_NAMESPACE=ailab-safevlagent rjob stop $n >/dev/null 2>&1; return 1; fi
  if [ "$state" = Failed ] || [ "$state" = Stopped ] || [ "$state" = Succeeded ]; then
    a=$(cat $ST/$key.attempts 2>/dev/null || echo 0)
    [ $a -ge 3 ] && { [ -f $ST/$key.dead ] || { log "$n ended $state - out of retries"; touch $ST/$key.dead; }; return 0; }
    echo $((a + 1)) > $ST/$key.attempts; log "$n ended $state - resubmitting as $(cur $key)"; submit $(cur $key) $g $c $m "$@"; fi
  return 1
}
egx_done() { [ -f $X/results/$TAG/scores.json ]; }
es_done() { [ -f $S/results/$TAG/scores.json ]; }
etq_shard() { local j=$1 sp ds n; for sp in direct indirect; do ds=etq_${sp}_s${j}of4; n=$(wc -l < $Q/data/$ds.jsonl)
  [ -f $Q/runs/$TAG/$ds/generated_predictions.jsonl ] && [ "$(wc -l < $Q/runs/$TAG/$ds/generated_predictions.jsonl)" -eq "$n" ] || return 1; done; }
etq_all() { local j; for j in 0 1 2 3; do etq_shard $j || return 1; done; }
etq_scored() { [ -f $Q/results/$TAG/scores.json ]; }
eab_pair() { local k; for k in $((2*$1)) $((2*$1+1)); do [ -f $BI/runs/$TAG/egoavu_bench_eval_s${k}of8/collect_stats.json ] || return 1; done; }
eab_all() { local j; for j in 0 1 2 3; do eab_pair $j || return 1; done; }
eab_judged() { [ -f $BI/results_h_fft/summary.json ]; }
eo_pair() { local k; for k in $((2*$1)) $((2*$1+1)); do [ -f $E/preds/$TAG/shard$k.jsonl ] && grep -q "DONE shard $k/8" $E/fft_epoch2/logs/job_s${k}of8_*.log 2>/dev/null || return 1; done; }
eo_all() { local j; for j in 0 1 2 3; do eo_pair $j || return 1; done; }
eo_judged() { grep -qh "egoOmni $TAG scored" $T/logs/egoomni_judge_*.log 2>/dev/null; }
loaded() {   # 0 = a smoke run loaded the checkpoint cleanly, 1 = not yet, 2 = bad load
  local f; for f in $X/logs/${TAG}__*.log $S/logs/${TAG}__*.log; do
    [ -f "$f" ] || continue
    grep -qE "newly initialized|were not used when initializing|size mismatch|Error\(s\) in loading" "$f" && { log "bad load in $f"; return 2; }
    grep -q "all params:" "$f" && { log "clean load: $(grep -h 'all params:' "$f" | tail -1 | cut -c1-160)"; return 0; }
  done; return 1; }
log start
until [ -f $L/egoavu/models/groo_ckpt_fft_epoch2/MODEL_READY ]; do sleep 60; done
log "model ready: $M"
phase=smoke
while :; do
  pending=0
  tend egx-$J 1 24 192000 egx_done -- bash $T/egocross_h.sh $TAG $AD || pending=$((pending + 1))
  tend es-$J 1 24 192000 es_done -- bash $T/egoschema_h.sh $TAG $AD || pending=$((pending + 1))
  if [ $phase = smoke ]; then
    loaded; rc=$?
    [ $rc -eq 2 ] && { log "SMOKE FAILED: checkpoint did not load cleanly - STOP"; exit 1; }
    [ $rc -eq 0 ] && { log "smoke ok - submitting EgoTaskQA / EgoAVU-Bench / egoOmni"; phase=all; }
    [ $phase = smoke ] && [ -f $ST/egx-$J.dead ] && [ -f $ST/es-$J.dead ] && { log "smoke jobs out of retries - STOP"; exit 1; }
  fi
  if [ $phase = all ]; then
    for j in 0 1 2 3; do
      tend etq-$J-p$j 1 24 192000 "etq_shard $j" -- bash $T/egotaskqa_h.sh $TAG $AD $j || pending=$((pending + 1))
      tend eab-$J-p$j 1 24 192000 "eab_pair $j" -- bash $T/bench2_h.sh $TAG $AD $((2*j)) $((2*j+1)) || pending=$((pending + 1))
      tend eo-$J-p$j 1 24 192000 "eo_pair $j" -- bash $T/egoomni2_h.sh fft_epoch2 8 $((2*j)) $((2*j+1)) || pending=$((pending + 1))
    done
    if etq_all; then tend etq-score-$J 1 16 192000 etq_scored -- bash $T/egotaskqa_score_h.sh $TAG || pending=$((pending + 1))
    elif ! ls $ST/etq-$J-p*.dead >/dev/null 2>&1; then pending=$((pending + 1)); fi
    if eab_all; then tend eab-judge-$J 4 32 256000 eab_judged -- bash $T/eab_judge_h.sh --suffix _fft $TAG || pending=$((pending + 1))
    elif ! ls $ST/eab-$J-p*.dead >/dev/null 2>&1; then pending=$((pending + 1)); fi
    if eo_all; then tend eoj-$J 1 16 192000 eo_judged -- bash $T/egoomni_judge_h.sh $TAG || pending=$((pending + 1))
    elif ! ls $ST/eo-$J-p*.dead >/dev/null 2>&1; then pending=$((pending + 1)); fi
  else pending=$((pending + 1)); fi
  [ $pending -eq 0 ] && break
  sleep 120
done
log "ALL DONE"; ls $ST/*.dead 2>/dev/null && log "gave up on: $(ls $ST/*.dead | xargs -n1 basename | tr '\n' ' ')"
