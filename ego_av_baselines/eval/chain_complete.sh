#!/bin/bash
# Drive BENCH to completion for the given models (openvla): wait until the bench's media are complete, then repeatedly
# submit 1-GPU jobs for whatever is still unanswered (resume skips answered ids) until nothing is missing or 4 rounds pass.
# Round 1 uses the model's normal packing; later rounds run 1 process per GPU (the leftovers are usually OOMs on long clips).
# usage: chain_complete.sh BENCH MODEL [MODEL ...]
LY=/mnt/shared-storage-user/ai4good1-share/liyu; E=$LY/egobench_eval; BENCH=$1; shift
source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
log() { echo "[$(date '+%m-%d %T')] $*"; }
missing() {   # unanswered request ids for model $1
  python3 - "$1" <<PY
import json, glob, sys
m = sys.argv[1]
ids = [json.loads(l)["id"] for l in open("$E/requests/$BENCH.jsonl")]
ok = set()
for f in glob.glob(f"$E/preds/{m}/$BENCH/shard_*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if "pred" in r:
            ok.add(r["id"])
print(sum(1 for i in ids if i not in ok))
PY
}
running() { brainctl get rjobs -n ailab-safevlagent 2>/dev/null | awk -v p="$1" 'NR>1 && index($1, p) == 1 && $2 ~ /Running|Starting|Inqueue/' | wc -l; }
if [ "$BENCH" = egotom ]; then
  until [ "$(brainctl get rjobs -n ailab-safevlagent 2>/dev/null | awk '$1=="egotom-clips" {print $2}')" = Succeeded ]; do sleep 120; done
  log "egotom clips complete"
fi
while [ "$(running et3-salmonn2p)" != 0 ]; do sleep 120; done
cd $LY/tools/h
for round in 1 2 3 4; do
  any=0
  for m in "$@"; do
    n=$(missing $m); [ "$n" = 0 ] && continue
    any=1
    if [ $round = 1 ]; then
      case $m in qwen2vl|llavaov|egogpt) J=$(( (n + 599) / 600 )); P=3 ;; minicpmo) J=$(( (n + 199) / 200 )); P=1 ;; salmonn2p) J=$(( (n + 299) / 300 )); P=2 ;; esac
    else
      J=$(( (n + 149) / 150 )); P=1
    fi
    [ $J -gt 8 ] && J=8
    log "round $round: $m missing $n -> $J jobs x $P procs"
    for j in $(seq 0 $((J - 1))); do
      bash submit_h.sh fin$round-$BENCH-$m-$j safevlagent 1 -- bash -c \
        "bash $E/run_infer.sh $m $BENCH $((J * P)) $((j * P)) $P > $E/logs/job_fin$round-$BENCH-$m-$j.log 2>&1" 2>&1 | grep -o "created rjob_name: .*"
    done
  done
  [ $any = 0 ] && break
  sleep 180
  while [ "$(running fin$round-$BENCH)" != 0 ]; do sleep 120; done
done
for m in "$@"; do log "final: $m missing $(missing $m)"; done
log CHAIN_COMPLETE
