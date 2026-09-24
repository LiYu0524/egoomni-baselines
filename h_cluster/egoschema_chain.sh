#!/usr/bin/env bash
# openvla: EgoSchema Subset pipeline — media prep (CPU rjob) -> build data -> 4 inference jobs (base + 3 EgoAVU LoRAs).
L=/mnt/shared-storage-user/ai4good1-share/liyu; T=$L/tools/h; R=$L/egoschema_eval_h; REL=/ai4good1-shared/liyu/egoavu/release
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
log() { echo "[es-chain $(date -Is)] $*"; }
st() { brainctl get rjobs $1 -n ailab-safevlagent --no-headers 2>/dev/null | awk '{print $2}'; }
log start
[ -n "$(st es-prep-subset2)" ] || H_IMAGE=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab H_CPU=48 H_MEM=160000 \
  $T/submit_h.sh es-prep-subset2 safevlagent 0 -- bash $T/egoschema_prep_h.sh subset 14 | grep -E "created|rror"
until grep -q "egoschema prep done" $R/logs/prep_subset_*.log 2>/dev/null; do
  s=$(st es-prep-subset2); case $s in Failed|Stopped|Succeeded) grep -q "egoschema prep done" $R/logs/prep_subset_*.log 2>/dev/null || { log "prep ended $s without success - STOP"; exit 1; };; esac
  sleep 60; done
log "prep done: $(grep -h "^\[prep\]" $R/logs/prep_subset_*.log | tail -1)"
python3 $R/build_data.py || { log "build_data FAILED"; exit 1; }
for pair in "base:none" "r100k:$REL/EgoAVU-Qwen2.5-Omni-7B-LoRA-r100k" "r20k8g:$REL/EgoAVU-Qwen2.5-Omni-7B-LoRA-r20k" "r20k32g:$REL/EgoAVU-Qwen2.5-Omni-7B-LoRA-r20k/32gpu"; do
  tag=${pair%%:*}; ad=${pair#*:}
  [ -n "$(st es-$tag)" ] || H_CPU=24 H_MEM=192000 $T/submit_h.sh es-$tag safevlagent 1 -- bash $T/egoschema_h.sh $tag $ad | grep -E "created|rror"
done
log "4 inference jobs submitted"
until [ $(ls $R/results/*/scores.json 2>/dev/null | wc -l) -ge 4 ]; do
  for tag in base r100k r20k8g r20k32g; do s=$(st es-$tag); [ "$s" = Failed ] && [ ! -f $R/results/$tag/scores.json ] && [ ! -f $R/.retried_$tag ] && {
    touch $R/.retried_$tag; log "es-$tag failed - resubmitting once"; ad=$(grep -h "adapter_name_or_path" $R/configs/${tag}__*.yaml 2>/dev/null | head -1 | awk '{print $2}'); ad=${ad:-none}
    H_CPU=24 H_MEM=192000 $T/submit_h.sh es-$tag-r safevlagent 1 -- bash $T/egoschema_h.sh $tag $ad | grep -E "created|rror"; }; done
  sleep 120; done
log "EgoSchema DONE"; for f in $R/results/*/scores.json; do python3 -c "import json; s=json.load(open('$f')); print(s['tag'], s['acc_lmms_eval_parse'], s['acc_lenient_parse'])"; done
