#!/usr/bin/env bash
# openvla: complete the egoOmni matrix. Inference for base / ckpt_sft / ckpt_epoch2 (4 one-GPU jobs each, 2 shards of 8 per job);
# each model is judged as soon as its 8 shards are final; the four baselines with existing predictions are judged right away.
# Failed or stuck (pod-sandbox) jobs are resubmitted up to 3 times.  Log: tools/h/logs/egoomni_complete_chain.log
L=/mnt/shared-storage-user/ai4good1-share/liyu; T=$L/tools/h; E=$L/egoOmni_baselines/eval; ST=$T/state_complete; mkdir -p $ST
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
declare -A TAG=( [base]=qwen25omni7b_base [ckpt_sft]=ckpt_sft [ckpt_epoch2]=ckpt_epoch2 )
BASELINES="salmonn2plus_7b salmonn2plus_72b minicpmo_2_6_8b gemini_3_8_flash"
log() { echo "[complete $(date -Is)] $*"; }
st() { brainctl get rjobs $1 -n ailab-safevlagent --no-headers 2>/dev/null | awk '{print $2, $3}'; }
jn() { echo $1 | tr "_" "-"; }
age_min() { echo "$1" | awk '{t=0; s=$0; while (match(s, /[0-9]+[dhms]/)) {v=substr(s, RSTART, RLENGTH); n=substr(v, 1, length(v)-1)+0; u=substr(v, length(v));
  t += (u=="d") ? n*1440 : (u=="h") ? n*60 : (u=="m") ? n : 0; s=substr(s, RSTART+RLENGTH)} print t}'; }
cur() { local a=$(cat $ST/$1.attempts 2>/dev/null || echo 0); [ $a -eq 0 ] && echo $1 || echo $1-r$a; }
submit() { local n=$1 c=$2 m=$3; shift 3
  H_CPU=$c H_MEM=$m $T/submit_h.sh $n safevlagent 1 -- "$@" 2>&1 | grep -q "created rjob_name" && log "submitted $n" || log "SUBMIT FAILED $n"; }
# keep job KEY alive until DONE_CMD succeeds: KEY CPU MEM DONE_CMD -- cmd...
tend() { local k=$1 c=$2 m=$3 d=$4; shift 5
  eval "$d" && return 0
  local n=$(cur $k) s a; s=$(st $n); [ -z "$s" ] && { submit $n $c $m "$@"; return 1; }
  set -- $s "$@"; local state=$1 age=$2; shift 2
  if [ "$state" = Starting ] && [ "$(age_min $age)" -ge 20 ] && KUBEBRAIN_NAMESPACE=ailab-safevlagent timeout 60 rjob events $n 2>/dev/null | grep -q "Failed to create pod sandbox"; then
    log "$n stuck in Starting ($age, pod-sandbox errors) - stopping"; KUBEBRAIN_NAMESPACE=ailab-safevlagent rjob stop $n >/dev/null 2>&1; return 1; fi
  if [ "$state" = Failed ] || [ "$state" = Stopped ] || [ "$state" = Succeeded ]; then
    a=$(cat $ST/$k.attempts 2>/dev/null || echo 0)
    [ $a -ge 3 ] && { [ -f $ST/$k.dead ] || { log "$n ended $state - out of retries"; touch $ST/$k.dead; }; return 0; }
    echo $((a + 1)) > $ST/$k.attempts; log "$n ended $state - resubmitting as $(cur $k)"; submit $(cur $k) $c $m "$@"; fi
  return 1
}
judged() { grep -qh "egoOmni $1 scored" $T/logs/egoomni_judge_*.log 2>/dev/null; }
shards_done() { local inst=$1 t=${TAG[$1]} k; for k in 0 1 2 3 4 5 6 7; do [ -f $E/preds/$t/shard$k.jsonl ] && grep -q "DONE shard $k/8" $E/$inst/logs/job_s${k}of8_*.log 2>/dev/null || return 1; done; }
job_done() { local inst=$1 j=$2 t=${TAG[$1]} k; for k in $((2*j)) $((2*j+1)); do [ -f $E/preds/$t/shard$k.jsonl ] && grep -q "DONE shard $k/8" $E/$inst/logs/job_s${k}of8_*.log 2>/dev/null || return 1; done; }
log "start"
while :; do
  pending=0
  for b in $BASELINES; do tend eoj-$(jn $b) 16 192000 "judged $b" -- bash $T/egoomni_judge_h.sh $b || pending=$((pending + 1)); done
  for inst in base ckpt_sft ckpt_epoch2; do
    for j in 0 1 2 3; do tend eo-$(jn $inst)-p$j 24 192000 "job_done $inst $j" -- bash $T/egoomni2_h.sh $inst 8 $((2*j)) $((2*j+1)) || pending=$((pending + 1)); done
    if shards_done $inst; then tend eoj-$(jn ${TAG[$inst]}) 16 192000 "judged ${TAG[$inst]}" -- bash $T/egoomni_judge_h.sh ${TAG[$inst]} || pending=$((pending + 1)); elif ! ls $ST/eo-$(jn $inst)-p*.dead >/dev/null 2>&1; then pending=$((pending + 1)); fi
  done
  [ $pending -eq 0 ] && break
  sleep 120
done
log "ALL DONE"; ls $ST/*.dead 2>/dev/null && log "some jobs gave up: $(ls $ST/*.dead | xargs -n1 basename | tr '\n' ' ')"
