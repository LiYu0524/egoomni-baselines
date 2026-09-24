#!/usr/bin/env bash
# Conductor on openvla (all jobs go to safevlagent_gpu; the ai4good1 group's GPUs were reclaimed): chains the H eval pipeline so no stage waits on a human.
#  EAB (EgoAVU-Bench): 12 inference jobs (1 H200 each, 2 procs) -> all 24 shards collected -> official judge job (4 GPU)
#  egoOmni: HF->s3 mirror -> media prep job (CPU) -> 12 inference jobs (1 H200 each, 2 shards of 8) -> judge + score job (1 GPU)
# Failed inference jobs are resubmitted once (name suffix -r). Log: tools/h/logs/conductor.log
L=/mnt/shared-storage-user/ai4good1-share/liyu; T=$L/tools/h; BI=$L/egoavu/bench_infer; E=$L/egoOmni_baselines/eval
R=/ai4good1-shared/liyu/egoavu/release; TAGS="r20k32g r100k r20k8g"
declare -A AD=( [r20k32g]=$R/EgoAVU-Qwen2.5-Omni-7B-LoRA-r20k/32gpu [r100k]=$R/EgoAVU-Qwen2.5-Omni-7B-LoRA-r100k [r20k8g]=$R/EgoAVU-Qwen2.5-Omni-7B-LoRA-r20k )
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
log() { echo "[conductor $(date -Is)] $*"; }
nsp() { [ "$1" = ai4good1 ] && echo ailab-ai4good1 || echo ailab-safevlagent; }
status() { brainctl get rjobs $1 -n $2 --no-headers 2>/dev/null | awk '{print $2}'; }
submit() {   # submit NAME PART GPUS CPU MEM cmd...
  local n=$1 p=$2 g=$3 c=$4 m=$5; shift 5
  if H_CPU=$c H_MEM=$m $T/submit_h.sh $n $p $g -- "$@" 2>&1 | grep -q "created rjob_name"; then log "submitted $n ($p, ${g} GPU)"; else log "SUBMIT FAILED $n"; fi
}
# attempt n of a job runs as NAME (0), NAME-r (1), NAME-r2 (2), NAME-r3 (3); infra failures (pod sandbox/network) are random per node
cur_name() { local a=$(cat $T/state/$1.attempts 2>/dev/null || { [ -f $T/state/$1.retry ] && echo 1 || echo 0; }); case $a in 0) echo $1;; 1) echo $1-r;; *) echo $1-r$a;; esac; }
submit_once() {   # submit_once NAME PART ... — skip if a job of that name already exists (conductor restarts)
  [ -n "$(status $1 $(nsp $2))" ] && return 0; submit "$@"
}
age_min() {   # kubectl-style age (45s, 27m, 2h3m, 1d2h) -> minutes
  echo "$1" | awk '{t=0; s=$0; while (match(s, /[0-9]+[dhms]/)) {v=substr(s, RSTART, RLENGTH); n=substr(v, 1, length(v)-1)+0; u=substr(v, length(v));
    t += (u=="d") ? n*1440 : (u=="h") ? n*60 : (u=="m") ? n : 0; s=substr(s, RSTART+RLENGTH)} print t}'
}
unstick() {   # stop a job stuck in Starting >20 min on a node whose pod sandbox/network cannot be created (platform fault)
  local n=$1 ns=$2 age
  age=$(brainctl get rjobs $n -n $ns --no-headers 2>/dev/null | awk '{print $3}')
  [ "$(age_min "$age")" -ge 20 ] || return 0
  KUBEBRAIN_NAMESPACE=$ns timeout 60 rjob events $n 2>/dev/null | grep -q "Failed to create pod sandbox" || return 0
  log "$n stuck in Starting for $age with pod-sandbox errors - stopping it so it gets resubmitted"
  KUBEBRAIN_NAMESPACE=$ns rjob stop $n >/dev/null 2>&1
}
# watch a set of jobs until DONE_FN(name) is true for all; resubmit a Failed/Stopped job up to 3 times. Registry: name part gpu cpu mem cmd...
watch_jobs() {   # watch_jobs REGISTRY DONE_FN LABEL
  local reg=$1 done_fn=$2 label=$3
  while :; do
    local pending=0
    while read -r n p g c m cmd; do
      [ -z "$n" ] && continue
      $done_fn $n && continue
      [ -f $T/state/$n.dead ] && continue
      pending=$((pending + 1))
      cur=$(cur_name $n); s=$(status $cur $(nsp $p))
      [ "$s" = Starting ] && unstick $cur $(nsp $p)
      if [ "$s" = Failed ] || [ "$s" = Stopped ] || { [ "$s" = Succeeded ] && ! $done_fn $n; }; then
        a=$(cat $T/state/$n.attempts 2>/dev/null || { [ -f $T/state/$n.retry ] && echo 1 || echo 0; })
        if [ $a -ge 3 ]; then log "$label: $cur ended $s - out of retries, giving up on it"; touch $T/state/$n.dead
        else a=$((a + 1)); echo $a > $T/state/$n.attempts; nn=$(cur_name $n)
             log "$label: $cur ended $s - resubmitting as $nn (retry $a/3)"; submit $nn $p $g $c $m $cmd; fi
      fi
    done < $reg
    [ $pending -eq 0 ] && return 0
    sleep 120
  done
}
mkdir -p $T/state

EAB=eab3                                   # job-name prefix of the current EAB inference round
eab_done() {   # $EAB-<tag>-p<j> is done when both of its 8-way shards are collected
  local tag=${1#$EAB-}; tag=${tag%-p*}; local j=${1##*-p}
  [ -f $BI/runs/$tag/egoavu_bench_eval_s$((2*j))of8/collect_stats.json ] && [ -f $BI/runs/$tag/egoavu_bench_eval_s$((2*j+1))of8/collect_stats.json ]
}
stage_eab() {
  # gate: the CPU preflight must show deepspeed finds CUDA 12.6 nvcc (the call that crashed the first EAB round)
  until grep -qh "installed_cuda_version -> " $T/logs/preflight_nvcc_cpu_*.log 2>/dev/null; do sleep 120; done
  grep -h "installed_cuda_version -> " $T/logs/preflight_nvcc_cpu_*.log | head -1 | grep -q "(12, 6)" \
    || { log "EAB: nvcc preflight did not report CUDA 12.6 - NOT submitting"; return 1; }
  log "EAB: nvcc preflight ok -> submitting inference"
  : > $T/state/eab.reg
  for tag in $TAGS; do for j in 0 1 2 3; do p=safevlagent
    echo "$EAB-$tag-p$j $p 1 24 192000 bash $T/bench2_h.sh $tag ${AD[$tag]} $((2*j)) $((2*j+1))" >> $T/state/eab.reg
    eab_done $EAB-$tag-p$j || submit_once $EAB-$tag-p$j $p 1 24 192000 bash $T/bench2_h.sh $tag ${AD[$tag]} $((2*j)) $((2*j+1)); done; done
  watch_jobs $T/state/eab.reg eab_done EAB
  ls $T/state/$EAB-*.dead >/dev/null 2>&1 && { log "EAB: some shards failed twice - judge NOT submitted"; return 1; }
  log "EAB: all 24 shards collected -> official judge"
  submit_once eab-judge-h safevlagent 4 32 512000 bash $T/eab_judge_h.sh $TAGS
  until grep -q "EAB judge DONE" $T/logs/eab_judge_*.log 2>/dev/null; do s=$(status eab-judge-h ailab-safevlagent); [ "$s" = Failed ] && { log "EAB judge job FAILED"; return 1; }; sleep 120; done
  log "EAB: judge + summary DONE -> $BI/results_h/SUMMARY.md"
}

eo_done() {    # eo-<inst>-p<j> is done when both of its shards wrote final preds
  local inst=${1#eo-}; inst=${inst%-p*}; local j=${1##*-p}
  [ -f $E/preds/egoavu_$inst/shard$((2*j)).jsonl ] && [ -f $E/preds/egoavu_$inst/shard$((2*j+1)).jsonl ] \
    && grep -q "DONE shard $((2*j))/8" $E/$inst/logs/job_s$((2*j))of8_*.log 2>/dev/null && grep -q "DONE shard $((2*j+1))/8" $E/$inst/logs/job_s$((2*j+1))of8_*.log 2>/dev/null
}
stage_egoomni() {
  MIR="^/root/miniconda3/bin/python mirror_egoomni_hf_to_s3.py"
  # this script unsets proxies for rjob; the HF mirror needs the PJLab proxy (object storage stays direct via .pjlab.org.cn)
  PJ_PROXY=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128
  PJ_NOPROXY=localhost,127.0.0.1,10.0.0.0/8,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.h.pjlab.org.cn,httpproxy-headless.kubebrain.svc.pjlab.local
  for attempt in 1 2 3 4 5; do   # 1) mirror HF -> s3; exactly one mirror process; a run with failures is rerun (it resumes)
    until grep -q "^\[mirror\] finished: " $T/logs/egoomni_mirror.log 2>/dev/null; do
      [ "$(pgrep -fc "$MIR")" -gt 0 ] || { log "mirror not running and not finished - starting one"; (cd $T && setsid nohup env http_proxy=$PJ_PROXY https_proxy=$PJ_PROXY HTTP_PROXY=$PJ_PROXY HTTPS_PROXY=$PJ_PROXY no_proxy=$PJ_NOPROXY NO_PROXY=$PJ_NOPROXY /root/miniconda3/bin/python mirror_egoomni_hf_to_s3.py f0f73dfa21b4e3e628a4771424ce06cfd93e6a57 8 >> logs/egoomni_mirror.log 2>&1 < /dev/null &); }
      sleep 300; done
    f=$(grep "^\[mirror\] finished: " $T/logs/egoomni_mirror.log | tail -1); log "mirror: $f"
    sed -i "s/^\[mirror\] finished: /[mirror] run-ended: /" $T/logs/egoomni_mirror.log
    echo "$f" | grep -q ", 0 failed" && break
    [ $attempt -eq 5 ] && { log "mirror still incomplete after 5 runs - egoOmni stage STOPPED"; return 1; }
    log "mirror had failures - rerunning (run $((attempt + 1)))"
  done
  until [ -x $L/egoOmni_baselines/envs_h/ffmpeg/bin/ffmpeg ]; do log "waiting for the ffmpeg env"; sleep 300; done
  H_IMAGE=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab submit_once egoomni-prep-h3 safevlagent 0 48 160000 bash $T/prep_egoomni_media_h.sh      # 2) media prep (CPU)
  until grep -q "prep done" $T/logs/prep_egoomni_media_*.log 2>/dev/null; do
    s=$(status egoomni-prep-h3 ailab-safevlagent); { [ "$s" = Failed ] || [ "$s" = Stopped ] || [ "$s" = Succeeded ]; } && ! grep -q "prep done" $T/logs/prep_egoomni_media_*.log 2>/dev/null && { log "egoOmni media prep FAILED"; return 1; }
    sleep 60; done
  log "egoOmni media prep done"
  : > $T/state/eo.reg                                                                 # 3) inference
  for inst in $TAGS; do for j in 0 1 2 3; do p=safevlagent
    echo "eo-$inst-p$j $p 1 24 192000 bash $T/egoomni2_h.sh $inst 8 $((2*j)) $((2*j+1))" >> $T/state/eo.reg
    submit_once eo-$inst-p$j $p 1 24 192000 bash $T/egoomni2_h.sh $inst 8 $((2*j)) $((2*j+1)); done; done
  watch_jobs $T/state/eo.reg eo_done egoOmni
  ls $T/state/eo-*.dead >/dev/null 2>&1 && { log "egoOmni: some shards failed twice - judge NOT submitted"; return 1; }
  log "egoOmni: all shards done -> judge + score"                                    # 4) judge + score
  submit_once egoomni-judge-h safevlagent 1 16 192000 bash $T/egoomni_judge_h.sh egoavu_r20k32g egoavu_r100k egoavu_r20k8g
  until grep -q "egoOmni judge DONE" $T/logs/egoomni_judge_*.log 2>/dev/null; do s=$(status egoomni-judge-h ailab-safevlagent); [ "$s" = Failed ] && { log "egoOmni judge job FAILED"; return 1; }; sleep 120; done
  log "egoOmni: judge + scores DONE -> $E/results/ and $E/results_final/"
}

log "conductor start (pid $$)"
stage_eab & stage_egoomni & wait
log "conductor finished"
