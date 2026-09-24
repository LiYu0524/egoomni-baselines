#!/usr/bin/env python3
"""EgoSchema media prep (CPU rjob). For each video of the chosen split, stream its mp4 out of videos_chunked_0N.zip
(read-only geesefs mount of s3://safevlagent/liyu/benchmarks/egoschema) and make the same proxy the EgoAVU LoRAs were
trained on (egoavu/tools/make_proxies_v2.py): fps=2, ~262k px/frame, H.264 veryfast crf 23, GOP 20, + 16 kHz mono s16 FLAC
when the video has an audio stream. Output: proxy/<video_idx>.{mp4,flac} and meta/media_<split>.json.
usage: prep_media.py subset|mc WORKERS"""
import json, os, shutil, subprocess, sys, zipfile
from multiprocessing import Pool
M = "/mnt/egoschema_s3"; R = "/ai4good1-shared/liyu/egoschema_eval_h"
FF = "/ai4good1-shared/liyu/egoOmni_baselines/envs_h/ffmpeg/bin/ffmpeg"; FP = FF[:-6] + "ffprobe"
TMP = "/var/tmp/egoschema"; os.makedirs(TMP, exist_ok=True); os.makedirs(f"{R}/proxy", exist_ok=True)
split, W = sys.argv[1], int(sys.argv[2])
vids = sorted({x["video_idx"] for x in json.load(open(f"{R}/meta/{split}.json"))})
where = {}
for i in range(1, 6):
    with zipfile.ZipFile(f"{M}/videos_chunked_0{i}.zip") as z:
        for n in z.namelist():
            if n.endswith(".mp4"):
                where[os.path.basename(n)[:-4]] = (i, n)
missing = [v for v in vids if v not in where]
assert not missing, f"{len(missing)} videos not found in the zips, e.g. {missing[:3]}"


def probe(p):
    o = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration:stream=codec_type,avg_frame_rate", "-of", "json", p],
                       capture_output=True, text=True).stdout
    j = json.loads(o or "{}"); st = j.get("streams", [])
    fr = next((s.get("avg_frame_rate") for s in st if s.get("codec_type") == "video"), "0/1") or "0/1"
    a, b = (fr.split("/") + ["1"])[:2]
    return float(j.get("format", {}).get("duration", 0) or 0), any(s.get("codec_type") == "audio" for s in st), (float(a) / float(b) if float(b) else 0.0)


def one(v):
    mp4, flac = f"{R}/proxy/{v}.mp4", f"{R}/proxy/{v}.flac"
    if os.path.exists(mp4):
        d, _, _ = probe(mp4)
        return v, {"proxy_duration": d, "has_audio": os.path.exists(flac), "cached": True}
    i, name = where[v]; tmp = f"{TMP}/{v}.mp4"
    try:
        with zipfile.ZipFile(f"{M}/videos_chunked_0{i}.zip") as z, z.open(name) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
        dur, has_a, fps = probe(tmp)
        sc = "scale=trunc(sqrt(262144*iw/ih)/2)*2:trunc(sqrt(262144*ih/iw)/2)*2:flags=bicubic"
        vf = sc if fps < 2.0 else "fps=2," + sc
        cmd = [FF, "-nostdin", "-v", "error", "-threads", "3", "-i", tmp, "-map", "0:v:0", "-vf", vf, "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "20", "-an", "-movflags", "+faststart",
               "-f", "mp4", "-y", mp4 + ".part"]
        if has_a:
            cmd += ["-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "flac", "-sample_fmt", "s16", "-f", "flac", "-y", flac + ".part"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return v, {"error": r.stderr[-300:]}
        os.replace(mp4 + ".part", mp4)
        if has_a:
            os.replace(flac + ".part", flac)
        pdur, _, pfps = probe(mp4)
        return v, {"duration": dur, "proxy_duration": pdur, "proxy_fps": pfps, "src_fps": fps, "has_audio": has_a, "zip": i}
    except Exception as e:
        return v, {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    with Pool(W) as p:
        res = dict(p.imap_unordered(one, vids, chunksize=1))
    errs = {v: r for v, r in res.items() if "error" in r}
    json.dump(res, open(f"{R}/meta/media_{split}.json", "w"), indent=1)
    print(f"[prep] {split}: {len(res)} videos, {len(errs)} errors, with audio {sum(bool(r.get('has_audio')) for r in res.values())}", list(errs.items())[:2])
    sys.exit(1 if errs else 0)
