#!/usr/bin/env python3
"""EgoToM-kit re-test on egoOmni: 2 fps proxies of every egoOmni test clip (original + restored_v3), same spec as the EgoSchema /
EgoTaskQA / EgoAVU proxies (tools make_proxies_v2: fps=2, ~262k px/frame, H.264 veryfast crf 23, GOP 20, yuv420p), video only
(audio comes from the 16 kHz FLACs in eval/r100k/audio16k). decord cannot decode the source clips; each clip is decoded once here
instead of once per question per model at inference time. Output mirrors the source tree under eval/kit_proxy2fps/.
usage: prep_egoomni_proxy2fps.py WORKERS   (CPU rjob with the egoOmni bucket mounted)"""
import glob, json, os, subprocess, sys
from multiprocessing import Pool
E = "/ai4good1-shared/liyu/egoOmni_baselines/eval"; SRC = "/ai4good1-shared/liyu/egoOmni/test/"; DST = f"{E}/kit_proxy2fps/"
FF = "/ai4good1-shared/liyu/egoOmni_baselines/envs_h/ffmpeg/bin/ffmpeg"; FP = FF.replace("bin/ffmpeg", "bin/ffprobe")
W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
clips = sorted({json.loads(l)["videos"][0].split("#t=")[0] for f in glob.glob(f"{E}/kit_base/data/gold_s*of16.jsonl") for l in open(f)})
assert clips and all(c.startswith(SRC) for c in clips), "unexpected clip paths"


def probe(p):
    o = subprocess.run([FP, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate:format=duration", "-of", "json", p],
                       capture_output=True, text=True).stdout
    j = json.loads(o or "{}"); fr = (j.get("streams") or [{}])[0].get("avg_frame_rate", "0/1") or "0/1"
    a, b = (fr.split("/") + ["1"])[:2]
    return float(j.get("format", {}).get("duration", 0) or 0), (float(a) / float(b) if float(b) else 0.0)


def one(src):
    dst = DST + src[len(SRC):]
    if os.path.exists(dst):
        return src, {"cached": True}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        dur, fps = probe(src)
        sc = "scale=trunc(sqrt(262144*iw/ih)/2)*2:trunc(sqrt(262144*ih/iw)/2)*2:flags=bicubic"
        vf = sc if 0 < fps < 2.0 else "fps=2," + sc
        r = subprocess.run([FF, "-nostdin", "-v", "error", "-threads", "3", "-i", src, "-map", "0:v:0", "-vf", vf, "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "20", "-an", "-movflags", "+faststart",
                            "-f", "mp4", "-y", dst + ".part"], capture_output=True, text=True)
        if r.returncode:
            return src, {"error": r.stderr[-300:]}
        os.replace(dst + ".part", dst)
        pdur, pfps = probe(dst)
        return src, {"duration": dur, "src_fps": fps, "proxy_duration": pdur, "proxy_fps": pfps}
    except Exception as e:
        return src, {"error": f"{type(e).__name__}: {str(e)[:300]}"}


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    with Pool(W) as p:
        res = dict(p.imap_unordered(one, clips, chunksize=2))
    errs = {k: v for k, v in res.items() if "error" in v}
    json.dump(res, open(f"{DST}/media.json", "w"), indent=1)
    bad = [k for k, v in res.items() if "proxy_duration" in v and abs(v["proxy_duration"] - v["duration"]) > 1.0]
    print(f"[proxy] {len(res)} clips, {len(errs)} errors, {len(bad)} with |proxy - source duration| > 1 s", list(errs.items())[:2], bad[:3])
    if not errs:
        open(f"{DST}/READY", "w").write("ok\n")
    sys.exit(1 if errs else 0)
