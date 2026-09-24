#!/usr/bin/env python3
"""EgoTaskQA media prep (CPU rjob). For every clip used by the direct + indirect TEST splits, stream
qa_videos/<interval>.mp4 out of qa_videos.zip (read-only geesefs mount of s3://safevlagent/liyu/benchmarks/egotaskqa/data)
and make the proxy format the EgoAVU LoRAs were trained on (egoavu/tools/make_proxies_v2.py): fps=2, ~262k px/frame,
H.264 veryfast crf 23, GOP 20, + 16 kHz mono s16 FLAC when the clip has an audio stream.
Proxies are named <interval with '|' -> '__'>.{mp4,flac}.  Output: proxy/, meta/media.json.   usage: prep_media.py WORKERS"""
import json, os, shutil, subprocess, sys, zipfile
from multiprocessing import Pool
M = "/mnt/egotaskqa_s3"; R = "/ai4good1-shared/liyu/egotaskqa_eval_h"
FF = "/shared/egoOmni_envs/minicpmo/bin/ffmpeg"; FP = "/shared/egoOmni_envs/minicpmo/bin/ffprobe"   # symlinks made by the job
TMP = "/var/tmp/egotaskqa"; os.makedirs(TMP, exist_ok=True); os.makedirs(f"{R}/proxy", exist_ok=True)
W = int(sys.argv[1]) if len(sys.argv) > 1 else 14
ivs = sorted({q["interval"] for s in ("direct", "indirect") for q in json.load(open(f"{R}/meta/qa_{s}_test_qas.json"))})
with zipfile.ZipFile(f"{M}/qa_videos.zip") as z:
    names = {os.path.splitext(os.path.basename(n))[0]: n for n in z.namelist() if n.endswith(".mp4")}
missing = [v for v in ivs if v not in names]
assert not missing, f"{len(missing)} test clips not in the zip, e.g. {missing[:3]}"
safe = lambda iv: iv.replace("|", "__")


def probe(p):
    o = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration:stream=codec_type,avg_frame_rate", "-of", "json", p],
                       capture_output=True, text=True).stdout
    j = json.loads(o or "{}"); st = j.get("streams", [])
    fr = next((s.get("avg_frame_rate") for s in st if s.get("codec_type") == "video"), "0/1") or "0/1"
    a, b = (fr.split("/") + ["1"])[:2]
    return float(j.get("format", {}).get("duration", 0) or 0), any(s.get("codec_type") == "audio" for s in st), (float(a) / float(b) if float(b) else 0.0)


def one(iv):
    mp4, flac = f"{R}/proxy/{safe(iv)}.mp4", f"{R}/proxy/{safe(iv)}.flac"
    if os.path.exists(mp4):
        d, _, _ = probe(mp4)
        return iv, {"proxy_duration": d, "has_audio": os.path.exists(flac), "cached": True}
    tmp = f"{TMP}/{safe(iv)}.mp4"
    try:
        with zipfile.ZipFile(f"{M}/qa_videos.zip") as z, z.open(names[iv]) as src, open(tmp, "wb") as dst:
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
            return iv, {"error": r.stderr[-300:]}
        os.replace(mp4 + ".part", mp4)
        if has_a:
            os.replace(flac + ".part", flac)
        pdur, _, _ = probe(mp4)
        return iv, {"duration": dur, "proxy_duration": pdur, "src_fps": fps, "has_audio": has_a}
    except Exception as e:
        return iv, {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    with Pool(W) as p:
        res = dict(p.imap_unordered(one, ivs, chunksize=4))
    errs = {v: r for v, r in res.items() if "error" in r}
    json.dump(res, open(f"{R}/meta/media.json", "w"), indent=1)
    durs = sorted(r["proxy_duration"] for r in res.values() if "proxy_duration" in r)
    print(f"[prep] {len(res)} clips, {len(errs)} errors, with audio {sum(bool(r.get('has_audio')) for r in res.values())}, "
          f"duration s min/median/max {durs[0]:.1f}/{durs[len(durs)//2]:.1f}/{durs[-1]:.1f}", list(errs.items())[:2])
    sys.exit(1 if errs else 0)
