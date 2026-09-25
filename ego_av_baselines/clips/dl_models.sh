#!/bin/bash
# Download benchmark baseline weights to gpfs (openvla; ModelScope via the kubebrain proxy).
LY=/mnt/shared-storage-user/ai4good1-share/liyu; M=$LY/models_bench; PY=$LY/tools/egoclip_env/bin/python
mkdir -p $M
export https_proxy=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 http_proxy=$https_proxy
for spec in Qwen/Qwen2-VL-7B-Instruct llava-hf/llava-onevision-qwen2-7b-ov-hf OpenBMB/MiniCPM-o-2_6 lmms-lab/EgoGPT-7b-EgoIT-EgoLife; do
  name=${spec#*/}
  echo "[$(date +%T)] start $spec"
  $PY -m modelscope.cli.cli download --model $spec --local_dir $M/$name --max-workers 8 > $M/.dl_$name.log 2>&1 \
    && echo "[$(date +%T)] done $spec $(du -sh $M/$name | cut -f1)" || echo "[$(date +%T)] FAILED $spec (see .dl_$name.log)"
done
echo ALL_MS_DONE
