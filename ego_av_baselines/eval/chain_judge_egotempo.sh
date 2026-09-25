#!/bin/bash
# Wait for the EgoTempo inference jobs of the given models to finish, then LLM-judge them (openvla).
# usage: chain_judge_egotempo.sh JOBPREFIX MODEL [MODEL ...]
LY=/mnt/shared-storage-user/ai4good1-share/liyu; E=$LY/egobench_eval; PFX=$1; shift
source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
while brainctl get rjobs -n ailab-safevlagent 2>/dev/null | awk -v p="^$PFX" 'NR>1 && $1 ~ p && $2 ~ /Running|Starting|Inqueue/' | grep -q .; do sleep 120; done
cd $E
for m in "$@"; do
  echo "[$(date +%T)] judging egotempo $m"
  /root/miniconda3/bin/python judge_openended.py egotempo $m --workers 16 2>&1 | tail -n 2
done
echo CHAIN_DONE
