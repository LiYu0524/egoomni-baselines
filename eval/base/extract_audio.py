#!/usr/bin/env python
"""16 kHz mono FLAC of every egoOmni clip that has an audio stream (original + restored_v3) — the audio format the
EgoAVU LoRA was trained on (proxy .flac 16 kHz mono). → eval/base/audio16k/<clip_rel>.flac"""
import json, os, subprocess, sys
from multiprocessing import Pool
FF = "/shared/egoOmni_envs/minicpmo/bin/ffmpeg"
D = "/ai4good1-shared/liyu/egoOmni"; OUT = "/ai4good1-shared/liyu/egoOmni_baselines/eval/base/audio16k"
metas = [json.load(open("/ai4good1-shared/liyu/egoOmni_baselines/eval/clips_meta.json")),
         json.load(open("/ai4good1-shared/liyu/egoOmni_baselines/eval/clips_meta_restored_v3.json"))]
todo = sorted({k for m in metas for k, v in m.items() if v.get("has_audio")})
def one(rel):
    out = os.path.join(OUT, rel[:-4] + ".flac")
    if os.path.exists(out): return rel, "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    r = subprocess.run([FF, "-v", "error", "-y", "-i", os.path.join(D, rel), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", out + ".tmp.flac"], capture_output=True, text=True)
    if r.returncode: return rel, "ERR " + r.stderr[-200:]
    os.replace(out + ".tmp.flac", out); return rel, "ok"
if __name__ == "__main__":
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 48) as p:
        res = p.map(one, todo, chunksize=4)
    errs = [r for r in res if r[1].startswith("ERR")]
    print(f"clips with audio: {len(todo)}  errors: {len(errs)}", errs[:3])
