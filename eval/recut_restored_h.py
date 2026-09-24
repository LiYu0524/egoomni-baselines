#!/usr/bin/env python
"""Re-create test/clips_restored_v3/ on H from the P-era manifest (P's source jsonl test/qa_v2_restored.jsonl is gone).
Uses prepare_restored.py's own cut functions with the choices the manifest recorded (method, window, sources, crop, size),
then checks duration / resolution / audio against clips_meta_restored_v3.json and re-runs the full_source pixel check.
usage: recut_restored_h.py [WORKERS]"""
import json, os, sys, subprocess
from multiprocessing import Pool
E = "/ai4good1-shared/liyu/egoOmni_baselines/eval"
sys.path.insert(0, E)
import prepare_restored as pr                     # FFMPEG env var selects the ffmpeg binary
from egoomni_eval import DATA_DIR

man = json.load(open(f"{E}/data/restored_v3/restored_v3_manifest.json"))
meta = json.load(open(f"{E}/clips_meta_restored_v3.json"))
qa = json.load(open(os.path.join(DATA_DIR, "test", "qa.json")))
clips = {}
for o in qa:
    clips.setdefault(o["video_id"], {})[o["clip_path"]] = (o["source_clip_interval"]["start_sec"], o["source_clip_interval"]["end_sec"])


def probe(p):
    o = subprocess.run([pr.FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", p],
                       capture_output=True, text=True).stdout
    j = json.loads(o or "{}"); v = [s for s in j.get("streams", []) if s.get("codec_type") == "video"]
    return (float(j.get("format", {}).get("duration", 0)), v[0]["width"] if v else 0, v[0]["height"] if v else 0,
            any(s.get("codec_type") == "audio" for s in j.get("streams", [])))


def one(m):
    s, e = m["window"]
    rel = f"test/clips_restored_v3/{m['video_id']}/{m['key']}.mp4"; out = os.path.join(DATA_DIR, rel)
    vc = sorted(clips[m["video_id"]].items(), key=lambda kv: kv[1])
    try:
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True); tmp = out + ".tmp.mp4"
            if m["method"] == "single_clip":
                cp = m["from_clips"][0]; a, b = clips[m["video_id"]][cp]
                pr.cut_one(os.path.join(DATA_DIR, cp), max(0.0, s - a), min(b - a, e - a), tmp)
            elif m["method"] == "full_source":
                pr.cut_full(os.path.join(DATA_DIR, m["source_video"]), s, e, m["crop_left_half"], m["output_size"], tmp)
            else:                                   # greedy chain, exactly as prepare_restored.py builds it
                t, chain = s, []
                while t < e - pr.EPS:
                    nxt = max(((cp, a, b) for cp, (a, b) in vc if a <= t + pr.EPS and b > t + pr.EPS), key=lambda x: x[2], default=None)
                    if nxt is None:
                        break
                    chain.append((os.path.join(DATA_DIR, nxt[0]), t - nxt[1], min(nxt[2], e) - nxt[1])); t = min(nxt[2], e)
                assert [os.path.relpath(p[0], DATA_DIR) for p in chain] == m["from_clips"], "stitch chain differs from manifest"
                size = max((pr.vinfo(p[0]) for p in chain), key=lambda wh: wh[0] * wh[1])
                pr.cut_stitched(chain, tmp, size)
            os.replace(tmp, out)
        dur, w, h, aud = probe(out); ref = meta[rel]
        bad = []
        if abs(dur - ref["duration"]) > 0.15: bad.append(f"dur {dur:.3f} vs {ref['duration']:.3f}")
        if (w, h) != (ref["width"], ref["height"]): bad.append(f"size {w}x{h} vs {ref['width']}x{ref['height']}")
        if aud != ref["has_audio"]: bad.append(f"audio {aud} vs {ref['has_audio']}")
        px = None
        if m["method"] == "full_source" and "pixel_check_vs_clip" in m:
            pc = m["pixel_check_vs_clip"]; a = clips[m["video_id"]][pc["clip"]][0]
            g1, g2 = pr.gray(out, pc["t_src"] - s), pr.gray(os.path.join(DATA_DIR, pc["clip"]), pc["t_src"] - a)
            px = (round(float(abs(g1 - g2).mean()), 2) if g1 is not None and g2 is not None else None, pc["mean_abs_diff_0_255"])
        return {"rel": rel, "method": m["method"], "ok": not bad, "bad": bad, "pixel_check_now_vs_P": px}
    except Exception as ex:
        return {"rel": rel, "method": m["method"], "ok": False, "bad": [f"{type(ex).__name__}: {str(ex)[:200]}"]}


if __name__ == "__main__":
    todo = [m for m in man if m["status"] == "cut"]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 32) as p:
        res = p.map(one, todo, chunksize=1)
    json.dump(res, open(f"{E}/data/restored_v3/recut_h_report.json", "w"), indent=1)
    nbad = [r for r in res if not r["ok"]]
    print(f"[recut] {len(res)} restored clips, {len(res) - len(nbad)} match P metadata, {len(nbad)} mismatched")
    for r in nbad[:10]: print("  BAD", r)
    print("  pixel checks (now, P):", [r["pixel_check_now_vs_P"] for r in res if r.get("pixel_check_now_vs_P")])
    sys.exit(1 if nbad else 0)
