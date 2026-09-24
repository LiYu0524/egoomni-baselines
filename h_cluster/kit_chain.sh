#!/usr/bin/env bash
# openvla: EgoToM-kit-protocol re-test of base / ckpt_epoch2 (= groo ckpt_lora_epoch2) / ckpt_fft_epoch2 on EgoAVU-Bench (30 shards,
# 3 processes per card) and egoOmni (16 shards, 4 per card), 1-GPU jobs at the per-GPU CPU/RAM share (8 CPU / 64 GB). Each model
# is judged as soon as its shards are complete: EgoAVU-Bench with the official judge into release_h_kit_<m> / judge_h_kit_<m> /
# results_h_kit_<m> (4 GPUs), egoOmni with Qwen3-32B (1 GPU). Failed or stuck (pod-sandbox) jobs are resubmitted up to 3 times;
# inference resumes per row.   Log: tools/h/logs/kit_chain.log
# (egoOmni jobs are named kit-eo2-*: the first kit-eo-* round failed before any row because the kit instances had no data;
# kit-eo2-* were OOM-killed: decord cannot decode egoOmni source clips and qwen_omni_utils fell back to torchvision.io.read_video,
# which loads whole clips into RAM -> nat_predict_lf.py now falls back to a PyAV reader with the same frame selection)
L=/mnt/shared-storage-user/ai4good1-share/liyu; T=$L/tools/h; ST=$T/state_kit; mkdir -p $ST
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
MODELS="ckpt_epoch2 ckpt_fft_epoch2 base"
EAB_K=30; EAB_PER=3; EO_K=16; EO_PER=4
BI=$L/egoavu/bench_infer; E=$L/egoOmni_baselines/eval
log() { echo "[kit $(date -Is)] $*"; }
st() { brainctl get rjobs $1 -n ailab-safevlagent --no-headers 2>/dev/null | awk '{print $2, $3}'; }
jn() { echo $1 | tr "_" "-"; }
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
eab_done() { local m=$1 s; shift; for s in "$@"; do [ -f $BI/runs/kit_$m/egoavu_bench_eval_s${s}of$EAB_K/collect_stats.json ] || return 1; done; }
eab_all() { eab_done $1 $(seq 0 $((EAB_K - 1))); }
eab_judged() { [ -f $BI/results_h_kit_$1/summary.json ]; }
eo_done() { local m=$1 s; shift; for s in "$@"; do
  [ -f $E/preds/kit_$m/shard$s.jsonl ] && grep -q "DONE shard $s/$EO_K" $E/kit_$m/logs/job_s${s}of${EO_K}_*.log 2>/dev/null || return 1; done; }
eo_all() { eo_done $1 $(seq 0 $((EO_K - 1))); }
eo_judged() { grep -qh "egoOmni kit_$1 scored" $T/logs/egoomni_judge_*.log 2>/dev/null; }
log "start (models: $MODELS; EgoAVU-Bench $EAB_K shards / $EAB_PER per card, egoOmni $EO_K shards / $EO_PER per card)"
while :; do
  pending=0
  for m in ckpt_epoch2 ckpt_fft_epoch2; do   # base: its tail runs as kit-eabrest-base-g* (rebalanced, submitted by hand)
    for ((j = 0; j < EAB_K / EAB_PER; j++)); do
      sh=$(seq $((j * EAB_PER)) $((j * EAB_PER + EAB_PER - 1)) | tr '\n' ' ')
      tend kit-eab-$(jn $m)-g$j 1 8 64000 "eab_done $m $sh" -- bash $T/kit_job_h.sh $m eab $EAB_K $sh || pending=$((pending + 1))
    done
  done
  # egoOmni: only once the 2 fps proxies exist and no HOLD_EGOOMNI flag is set (user frees cards for others while it is set)
  if false; then   # 2026-09-25 04:07 user: egoOmni is not ours to test -> the kit-protocol egoOmni re-test is dropped
  for m in $MODELS; do
    for ((j = 0; j < EO_K / EO_PER; j++)); do
      sh=$(seq $((j * EO_PER)) $((j * EO_PER + EO_PER - 1)) | tr '\n' ' ')
      tend kit-eo4-$(jn $m)-g$j 1 8 64000 "eo_done $m $sh" -- bash $T/kit_job_h.sh $m egoomni $EO_K $sh || pending=$((pending + 1))
    done
  done
  fi
  for m in $MODELS; do
    if eab_all $m; then tend kit-eabj-$(jn $m) 4 32 256000 "eab_judged $m" -- bash $T/eab_judge_h.sh --suffix _kit_$m --shards $EAB_K kit_$m || pending=$((pending + 1))
    else pending=$((pending + 1)); fi   # base: its tail runs as kit-eabrest-base-g* jobs (submitted by hand), merged by finalize_kit_rest.py
  done
  [ $pending -eq 0 ] && break
  sleep 30   # the EgoAVU-Bench judges must finish before 05:00
done
log "ALL DONE"; ls $ST/*.dead 2>/dev/null && log "gave up on: $(ls $ST/*.dead | xargs -n1 basename | tr '\n' ' ')"
