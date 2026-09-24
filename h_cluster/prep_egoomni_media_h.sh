#!/usr/bin/env bash
# CPU rjob: egoOmni media prep on H — re-cut the restored_v3 clips, then 16 kHz mono FLAC for every clip with audio (audio16k).
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee $LY/tools/h/logs/prep_egoomni_media_$(hostname).log) 2>&1
h_mount_egoomni || exit 1
FFENV=$LY/egoOmni_baselines/envs_h/ffmpeg; export FFMPEG=/shared/egoOmni_envs/minicpmo/bin/ffmpeg   # symlink below; prepare_restored derives ffprobe via str.replace("ffmpeg","ffprobe"), so the env dir name must not contain "ffmpeg"
mkdir -p /shared/egoOmni_envs/minicpmo/bin                 # extract_audio.py's hard-coded P ffmpeg path
ln -sf $FFENV/bin/ffmpeg /shared/egoOmni_envs/minicpmo/bin/ffmpeg; ln -sf $FFENV/bin/ffprobe /shared/egoOmni_envs/minicpmo/bin/ffprobe
PY=$LY/egoOmni_baselines/envs_h/judge/bin/python        # has numpy (pixel check)
E=/ai4good1-shared/liyu/egoOmni_baselines/eval; cd $E
h_log "recut restored_v3"; $PY recut_restored_h.py 24 || { h_log "recut FAILED"; exit 1; }
h_log "extract audio16k"; $PY r100k/extract_audio.py 40 | tee /tmp/extract_audio.out
grep -q "errors: 0 " /tmp/extract_audio.out || { h_log "extract_audio had errors - FAILED"; exit 1; }
n=$(find $E/r100k/audio16k -name "*.flac" | wc -l); h_log "audio16k files: $n"
h_log "prep done"
