#!/usr/bin/env bash
# Snapshot the NVMe runtime envs into single tarballs on the PVC (fast: one big sequential file each).
set -euo pipefail; source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
for e in "$@"; do
  [ -d $EGO_ENVS/$e ] || { echo "no env $e"; exit 1; }
  echo "[$(date -Is)] packing $e ..."; tar -C $EGO_ENVS -cf $EGO_ROOT/envs/$e.tar.tmp $e && mv $EGO_ROOT/envs/$e.tar.tmp $EGO_ROOT/envs/$e.tar
  echo "[$(date -Is)] $EGO_ROOT/envs/$e.tar ($(du -sh $EGO_ROOT/envs/$e.tar | cut -f1))"
done
