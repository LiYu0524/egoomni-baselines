#!/usr/bin/env bash
# openvla: EgoTaskQA for six models — wait for media prep -> build data -> 4 one-GPU inference jobs per model -> one scoring job
# per model as soon as its 8 shard runs exist. Failed / stuck jobs are resubmitted up to 3 times.  Log: logs/egotaskqa_chain.log
L=/mnt/shared-storage-user/ai4good1-share/liyu; T=$L/tools/h; R=$L/egotaskqa_eval_h; ST=$T/state_etq; mkdir -p $ST
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
REL=/ai4good1-shared/liyu/egoavu/release; AD=/ai4good1-shared/liyu/egoavu/bench_infer/adapters
declare -A ADP=( [base]=none [r100k]=$REL/EgoAVU-Qwen2.5-Omni-7B-LoRA-r100k [r20k8g]=$REL/EgoAVU-Qwen2.5-Omni-7B-LoRA-r20k
                 [r20k32g]=$REL/EgoAVU-Qwen2.5-Omni-7B-LoRA-r20k/32gpu [ckpt_sft]=$AD/ckpt_sft [ckpt_epoch2]=$AD/ckpt_epoch2 )
MODELS="base r20k8g r20k32g r100k ckpt_sft ckpt_epoch2"
log() { echo "[etq $(date -Is)] $*"; }
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
shard_done() { local tag=$1 j=$2 sp ds n; for sp in direct indirect; do ds=etq_${sp}_s${j}of4; n=$(wc -l < $R/data/$ds.jsonl)
  [ -f $R/runs/$tag/$ds/generated_predictions.jsonl ] && [ "$(wc -l < $R/runs/$tag/$ds/generated_predictions.jsonl)" -eq "$n" ] || return 1; done; }
model_done() { local tag=$1 j; for j in 0 1 2 3; do shard_done $tag $j || return 1; done; }
scored() { [ -f $R/results/$1/scores.json ]; }
log start
until grep -qh "egotaskqa prep done" $R/logs/prep_*.log 2>/dev/null; do
  s=$(st etq-prep); case ${s%% *} in Failed|Stopped) log "media prep ended ${s%% *} - STOP"; exit 1;; esac; sleep 60; done
log "prep: $(grep -h '^\[prep\]' $R/logs/prep_*.log | tail -1)"
[ -f $R/data/meta.json ] || python3 $R/build_data.py || { log "build_data FAILED - STOP"; exit 1; }
while :; do
  pending=0
  for t in $MODELS; do
    for j in 0 1 2 3; do tend etq-$(jn $t)-p$j 1 24 192000 "shard_done $t $j" -- bash $T/egotaskqa_h.sh $t ${ADP[$t]} $j || pending=$((pending + 1)); done
    if model_done $t; then tend etq-score-$(jn $t) 1 16 192000 "scored $t" -- bash $T/egotaskqa_score_h.sh $t || pending=$((pending + 1))
    elif ! ls $ST/etq-$(jn $t)-p*.dead >/dev/null 2>&1; then pending=$((pending + 1)); fi
  done
  [ $pending -eq 0 ] && break
  sleep 120
done
log "ALL DONE"; ls $ST/*.dead 2>/dev/null && log "gave up on: $(ls $ST/*.dead | xargs -n1 basename | tr '\n' ' ')"
