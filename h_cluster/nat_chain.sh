#!/usr/bin/env bash
# openvla: re-test the colleague's ModelScope models (ckpt_lora_epoch2 = ckpt_epoch2, ckpt_fft_epoch2) + the untuned base as the
# same-protocol reference with the colleague's EgoToM-kit protocol (tools/h/nat_infer.py) on EgoSchema, EgoTaskQA and EgoCross.
# Many small 1-GPU jobs (several processes per card, but only the per-GPU CPU/RAM share: 8 CPU / 64 GB for 4 processes —
# 24-CPU requests left GPUs idle on nodes whose CPUs were taken) to fill the free cards; a finished (model, benchmark) is converted to
# LLaMAFactory layout under tag kit_<model> and scored with the unchanged scorers (EgoTaskQA: 1-GPU Qwen3-32B judge job).
# Failed or stuck (pod-sandbox) jobs are resubmitted up to 3 times (inference resumes per question).  Log: logs/nat_chain.log
L=/mnt/shared-storage-user/ai4good1-share/liyu; T=$L/tools/h; ST=$T/state_nat; K=$L/natkit; mkdir -p $ST
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
MODELS="ckpt_epoch2 ckpt_fft_epoch2 base"
declare -A NS=( [egoschema]=${NS_ES:-16} [egotaskqa]=${NS_ETQ:-64} [egocross]=${NS_EGX:-8} )     # shards per benchmark
declare -A PER=( [egoschema]=${PER_ES:-2} [egotaskqa]=${PER_ETQ:-4} [egocross]=${PER_EGX:-4} )   # processes (shards) per card
declare -A KIT=( [egoschema]=egoschema_eval_h [egotaskqa]=egotaskqa_eval_h [egocross]=egocross_eval_h )
declare -A SHORT=( [egoschema]=es [egotaskqa]=etq [egocross]=egx )
BENCHES="egoschema egocross egotaskqa"
log() { echo "[nat $(date -Is)] $*"; }
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
job_done() {   # BENCH MODEL SHARD...  (against the status snapshot of this loop)
  local b=$1 m=$2 s; shift 2; for s in "$@"; do grep -qx "$b $m $s" $ST/complete.txt || return 1; done; }
model_done() { local b=$1 m=$2; [ "$(grep -c "^$b $m " $ST/complete.txt)" -eq "${NS[$b]}" ]; }
scored() { [ -f $L/${KIT[$1]}/results/kit_$2/scores.json ]; }
log "start (models: $MODELS; shards: es ${NS[egoschema]}/${PER[egoschema]} per card, etq ${NS[egotaskqa]}/${PER[egotaskqa]}, egx ${NS[egocross]}/${PER[egocross]})"
while :; do
  pending=0
  python3 $T/nat_status.py egoschema:${NS[egoschema]} egotaskqa:${NS[egotaskqa]} egocross:${NS[egocross]} > $ST/complete.txt.tmp && mv $ST/complete.txt.tmp $ST/complete.txt
  for b in $BENCHES; do          # benchmark-major: every model's EgoSchema jobs are queued before any EgoTaskQA job
    for m in $MODELS; do
      n=${NS[$b]}; p=${PER[$b]}
      for ((j = 0; j < n / p; j++)); do
        sh=$(seq $((j * p)) $((j * p + p - 1)) | tr '\n' ' ')
        tend nat-${SHORT[$b]}-$(jn $m)-g$j 1 $((2 * p)) $((16000 * p)) "job_done $b $m $sh" -- bash $T/nat_job_h.sh $m $b $n $sh || pending=$((pending + 1))
      done
      if model_done $b $m; then
        if ! scored $b $m; then
          if [ ! -f $L/${KIT[$b]}/runs/kit_$m/.converted ]; then
            python3 $T/nat_to_lf.py $b $m kit_$m && touch $L/${KIT[$b]}/runs/kit_$m/.converted || log "convert $b $m FAILED"
          fi
          case $b in
            egoschema|egocross) (cd /tmp && python3 $L/${KIT[$b]}/collect.py kit_$m > $ST/score_${b}_$m.log 2>&1) && log "$b kit_$m scored: $(tail -1 $ST/score_${b}_$m.log | cut -c1-200)" || { log "$b kit_$m scoring FAILED"; pending=$((pending + 1)); } ;;
            egotaskqa) tend nat-etq-score-$(jn $m) 1 16 192000 "scored egotaskqa $m" -- bash $T/egotaskqa_score_h.sh kit_$m || pending=$((pending + 1)) ;;
          esac
        fi
      elif ! ls $ST/nat-${SHORT[$b]}-$(jn $m)-g*.dead >/dev/null 2>&1; then pending=$((pending + 1)); fi
    done
  done
  [ $pending -eq 0 ] && break
  sleep 120
done
log "ALL DONE"; ls $ST/*.dead 2>/dev/null && log "gave up on: $(ls $ST/*.dead | xargs -n1 basename | tr '\n' ' ')"
