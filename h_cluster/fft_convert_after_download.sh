#!/usr/bin/env bash
# openvla: wait for the ckpt_fft_epoch2 download, then re-header it into the thinker view (convert_thinker_ckpt.py).
until grep -q "ALL DONE" /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/logs/fft_download.log; do
  pgrep -f "ms_dl.py groo_legend/ckpts ckpt_fft_epoch2" >/dev/null || { sleep 5; grep -q "ALL DONE" /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/logs/fft_download.log || { echo "download ended without ALL DONE"; echo "CONVERT EXIT 9"; exit 9; }; }
  sleep 20
done
echo "download complete $(date -Is)"
python3 /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/convert_thinker_ckpt.py /mnt/shared-storage-user/ai4good1-share/liyu/egoavu/models/groo_ckpt_fft_epoch2_src /mnt/shared-storage-user/ai4good1-share/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train /mnt/shared-storage-user/ai4good1-share/liyu/egoavu/models/groo_ckpt_fft_epoch2
rc=$?; echo "finished $(date -Is)"; echo "CONVERT EXIT $rc"
