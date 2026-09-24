#!/usr/bin/env bash
# CPU-only preflight for H eval jobs: s3 media mount, EgoAVU env restore + patches, media decode through the mount.
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee $LY/tools/h/logs/preflight_$(hostname).log) 2>&1
h_log "preflight start"
h_mount_s3 || exit 1
t=$(date +%s); tar -xf $LY/egoavu/backup/egoavu_nvme_20260921.tar -C / && h_log "env restored in $(( $(date +%s) - t )) s"
ENV=/shared/egoavu/env
$ENV/bin/python -c "import llamafactory, torch, peft, transformers; from llamafactory.data.mm_plugin import _egoavu_split_fragment; print('[pf] imports ok torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
$ENV/bin/python -c "import flash_attn; print('[pf] flash_attn', flash_attn.__version__)" 2>&1 | tail -1
row=$(head -1 $LY/egoavu/bench_infer/data/egoavu_bench_eval_s0of4.jsonl)
V=$(echo "$row" | $ENV/bin/python -c "import sys,json; print(json.load(sys.stdin)['videos'][0])")
A=$(echo "$row" | $ENV/bin/python -c "import sys,json; print(json.load(sys.stdin)['audios'][0])")
h_log "sample video=$V"
f=${V%%#*}; t=$(date +%s.%N); cat "$f" > /dev/null; h_log "read $(stat -Lc %s "$f") B of $(basename $f) in $(echo "$(date +%s.%N) - $t" | bc 2>/dev/null || echo ?) s"
$ENV/bin/python - "$V" "$A" <<'PY'
import sys, time
from llamafactory.data.mm_plugin import _egoavu_split_fragment
v, a = sys.argv[1], sys.argv[2]
print("[pf] fragment parse:", _egoavu_split_fragment(v), _egoavu_split_fragment(a))
import av
t = time.time(); c = av.open(_egoavu_split_fragment(v)[0]); s = c.streams.video[0]
n = 0
for fr in c.decode(s):
    n += 1
    if n >= 20: break
print(f"[pf] decoded {n} video frames {fr.width}x{fr.height} in {time.time()-t:.2f}s; fps={float(s.average_rate):.2f}")
import torchaudio
t = time.time(); w, sr = torchaudio.load(_egoavu_split_fragment(a)[0], frame_offset=0, num_frames=16000*30)
print(f"[pf] audio {tuple(w.shape)} sr={sr} in {time.time()-t:.2f}s via backend {torchaudio.list_audio_backends()}")
PY
h_log "preflight done"
