#!/usr/bin/env bash
# List shared model hubs on gpfs2-shared-public (not mounted on openvla).
L=/mnt/shared-storage-user/ai4good1-share/liyu/tools/h/logs/probe2_models.log; exec > >(tee $L) 2>&1
P=/mnt/shared-storage-gpfs2/gpfs2-shared-public
echo "[probe2] $(date -Is)"; ls $P | head -30; df -h $P | tail -1
for hub in $P/huggingface/hub $P/modelscope $P/models $P/hf_hub; do
  [ -d "$hub" ] || continue; echo "=== $hub ($(ls $hub | wc -l) entries)"
  ls $hub | grep -iE "qwen3-32b|qwen3-235b|qwen2\.5-omni|qwen3-30b|qwen2_5_omni|omni" | head -30
done
for m in "models--Qwen--Qwen3-32B" "models--Qwen--Qwen3-235B-A22B-Instruct-2507" "models--Qwen--Qwen2.5-Omni-7B"; do
  d=$P/huggingface/hub/$m; [ -d $d ] && { echo "--- $m"; ls $d/snapshots/; for s in $d/snapshots/*; do echo "$s: $(ls $s | wc -l) files, $(du -shL $s 2>/dev/null | cut -f1)"; done; }
done
echo "[probe2] done"
