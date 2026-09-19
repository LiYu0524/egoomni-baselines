#!/usr/bin/env bash
# After a container rebuild (/shared wiped): restore runtime envs from the PVC tarballs. ~minutes on NVMe.
set -euo pipefail; source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
mkdir -p $EGO_ENVS
for e in ${@:-videollama2 salmonn2plus}; do
  [ -x $EGO_ENVS/$e/bin/python ] && { echo "$e already present"; continue; }
  echo "[$(date -Is)] restoring $e ..."; tar -C $EGO_ENVS -xf $EGO_ROOT/envs/$e.tar; echo "[$(date -Is)] $e restored"
done
