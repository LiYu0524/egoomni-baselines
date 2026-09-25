#!/bin/bash
# EgoTempo judge calibration: the official judge gemini-1.5-flash is not on the gateway -> compare candidate judges on the
# two models with published EgoTempo numbers (Plizzari et al. 2025: Qwen2-VL-7B 26.10, LLaVA-OV-Qwen2-7B 23.30).
cd /mnt/shared-storage-user/ai4good1-share/liyu/egobench_eval
for j in gpt-4o-mini gemini-2.5-flash-lite gemini-2.5-flash-nothinking; do
  for m in qwen2vl llavaov; do
    r=$(/root/miniconda3/bin/python judge_openended.py egotempo $m --judge $j --workers 16 2>&1 | tail -n 1)
    echo "$j $m $r"
  done
done
echo CALIB_DONE
