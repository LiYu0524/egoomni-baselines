#!/usr/bin/env bash
# Check that NORMAL videos read through the s3 mount have real content (mean luma, audio RMS, full size).
source /mnt/shared-storage-user/ai4good1-share/liyu/tools/h/h_prelude.sh
exec > >(tee $LY/tools/h/logs/preflight2_$(hostname).log) 2>&1
h_mount_s3 || exit 1
tar -xf $LY/egoavu/backup/egoavu_nvme_20260921.tar -C /
/shared/egoavu/env/bin/python - <<'PY'
import av, numpy as np, soundfile as sf, os, time
P = "/ai4good1-shared/liyu/egoavu/proxy262k/test/"
for v in ["f7eb4ced-2b81-4820-8da8-87891645499c", "36ee5431-8d3f-4773-b158-2b42932666aa", "c0f3d0e4-9f20-4174-bb58-cb377df92255"]:
    t = time.time(); c = av.open(P + v + ".mp4"); s = c.streams.video[0]
    lum = []
    for i, fr in enumerate(c.decode(s)):
        if i % 20 == 0: lum.append(fr.to_ndarray(format="gray").mean())
        if len(lum) >= 8: break
    dur = float(s.duration * s.time_base) if s.duration else float(c.duration or 0) / 1e6
    a, sr = sf.read(P + v + ".flac", frames=16000 * 60)
    print(f"[pf2] {v[:8]} mp4 {os.path.getsize(P+v+'.mp4')/1e6:.2f} MB dur {dur:.1f}s {s.width}x{s.height} | mean luma of {len(lum)} frames: {np.round(lum,1).tolist()} | audio RMS {np.sqrt(np.mean(a**2)):.4f} | {time.time()-t:.1f}s")
PY
h_log "preflight2 done"
