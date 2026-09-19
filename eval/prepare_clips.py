#!/usr/bin/env python
"""ffprobe every clip referenced by qa.json once → eval/clips_meta.json {rel_path: {duration, has_audio, audio_codec, fps, width, height}}."""
import json
import os
import subprocess
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egoomni_eval import CLIPS_META, DATA_DIR, QA_PATH  # noqa: E402

FFPROBE = os.environ.get("FFPROBE", "/shared/egoOmni_envs/videollama2/bin/ffprobe")


def probe(rel):
    p = os.path.join(DATA_DIR, rel)
    try:
        out = subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", p], capture_output=True, text=True, timeout=120).stdout
        d = json.loads(out)
    except Exception as e:  # noqa: BLE001
        return rel, {"error": str(e)}
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = None
    if v and v.get("avg_frame_rate") and v["avg_frame_rate"] != "0/0":
        n, dn = v["avg_frame_rate"].split("/"); fps = round(float(n) / float(dn), 3) if float(dn) else None
    return rel, {"duration": float(d.get("format", {}).get("duration") or (v or {}).get("duration") or 0), "has_audio": a is not None,
                 "audio_codec": (a or {}).get("codec_name"), "video_codec": (v or {}).get("codec_name"), "fps": fps,
                 "width": (v or {}).get("width"), "height": (v or {}).get("height"), "size_bytes": int(d.get("format", {}).get("size") or 0)}


if __name__ == "__main__":
    items = json.load(open(QA_PATH))
    rels = sorted({it["clip_path"] for it in items})
    with Pool(32) as pool:
        meta = dict(pool.map(probe, rels, chunksize=8))
    json.dump(meta, open(CLIPS_META, "w"), indent=1)
    errs = [k for k, v in meta.items() if "error" in v]
    na = sum(1 for v in meta.values() if not v.get("has_audio"))
    print(f"clips={len(meta)} errors={len(errs)} no_audio={na} → {CLIPS_META}")
    if errs:
        print("errors:", errs[:5])
