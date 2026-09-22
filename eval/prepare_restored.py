#!/usr/bin/env python
"""Build the evaluable part of the 168 'restored_v3' items added in test.qa.jsonl (2026-09-22).

Those rows carry no clip — only a window `original_qa.video_clip_range` in the full Ego4D video. We cut each window
in this priority: (1) from one existing egoOmni clip of the same video (clip time = source time − source_clip_interval.start);
(2) from the full Ego4D video in <DATA_DIR>/ego4d_full/<video_id>.mp4 when available — cropped to the left half when the
source is a side-by-side frame (as the dataset clips of those videos are) and scaled to the largest dataset-clip resolution
of that video, then pixel-checked against an overlapping dataset clip; (3) by stitching overlapping clips; otherwise the item
is skipped and listed with the missing seconds. Re-encoded H.264 CRF 18 + AAC 128k (source clips are VP9/AAC).

Outputs (under the dataset dir): test/clips_restored_v3/<video_id>/<key>.mp4, test/qa_restored_v3.json (qa.json schema,
source_kind="restored_single"), test/restored_v3_manifest.json (all 168: status, clip window, skip reason).
  python prepare_restored.py test/qa_v2_restored.jsonl
"""
import json, os, subprocess, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egoomni_eval import DATA_DIR  # noqa: E402

FF = os.environ.get("FFMPEG", "/shared/egoOmni_envs/minicpmo/bin/ffmpeg")
FFPROBE = FF.replace("ffmpeg", "ffprobe")
EPS = 0.05
MODS = {("visual",): "V", ("audio", "visual"): "A+V", ("audio",): "A"}


def has_audio(p):
    o = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", p], capture_output=True, text=True).stdout
    return "audio" in o


def vinfo(p):
    o = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", p],
                       capture_output=True, text=True).stdout.strip().split(",")
    return int(o[0]), int(o[1])


def gray(p, t, vf_pre=""):
    import numpy as np
    r = subprocess.run([FF, "-v", "error", "-ss", f"{t:.3f}", "-i", p, "-frames:v", "1", "-vf", vf_pre + "scale=64:48,format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).astype(float) if len(r.stdout) == 64 * 48 else None


def enc_args(audio):
    a = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"]
    return a + (["-c:a", "aac", "-b:a", "128k"] if audio else ["-an"]) + ["-movflags", "+faststart"]


def cut_one(src, off0, off1, out):
    audio = has_audio(src)
    cmd = [FF, "-v", "error", "-y", "-i", src, "-ss", f"{off0:.3f}", "-to", f"{off1:.3f}"] + enc_args(audio) + [out]
    subprocess.run(cmd, check=True, capture_output=True)


def cut_full(src, s, e, crop_left, size, out):
    audio = has_audio(src)
    vf = ("crop=iw/2:ih:0:0," if crop_left else "") + f"scale={size[0]}:{size[1]},setsar=1"
    cmd = [FF, "-v", "error", "-y", "-ss", f"{s:.3f}", "-i", src, "-t", f"{e - s:.3f}", "-vf", vf] + enc_args(audio) + [out]
    subprocess.run(cmd, check=True, capture_output=True)


def cut_stitched(parts, out, size):
    """parts = [(src, off0, off1), ...] consecutive pieces of the window; clips of one video can differ in resolution."""
    audio = all(has_audio(p[0]) for p in parts)
    cmd = [FF, "-v", "error", "-y"]
    for src, _, _ in parts:
        cmd += ["-i", src]
    fc, labels = [], ""
    for i, (_, a, b) in enumerate(parts):
        fc.append(f"[{i}:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS,fps=30,scale={size[0]}:{size[1]},setsar=1,format=yuv420p[v{i}]")
        labels += f"[v{i}]"
        if audio:
            fc.append(f"[{i}:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,aresample=32000,aformat=channel_layouts=mono[a{i}]")
            labels += f"[a{i}]"
    fc.append(f"{labels}concat=n={len(parts)}:v=1:a={1 if audio else 0}[v]" + ("[a]" if audio else ""))
    cmd += ["-filter_complex", ";".join(fc), "-map", "[v]"] + (["-map", "[a]"] if audio else []) + enc_args(audio) + [out]
    subprocess.run(cmd, check=True, capture_output=True)


def main(src_jsonl):
    rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, src_jsonl)) if l.strip()]
    old = json.load(open(os.path.join(DATA_DIR, "test", "qa.json")))
    clips = defaultdict(dict)
    for o in old:
        clips[o["video_id"]][o["clip_path"]] = (o["source_clip_interval"]["start_sec"], o["source_clip_interval"]["end_sec"])
    restored = [r for r in rows if r.get("source_kind") is None]
    items, manifest = [], []
    for r in restored:
        q, key = r["original_qa"], r["restoration"]["source"]["key"]
        s, e = q["video_clip_range"]["start_sec"], q["video_clip_range"]["end_sec"]
        vc = sorted(clips[r["video_id"]].items(), key=lambda kv: kv[1])
        rel = f"test/clips_restored_v3/{r['video_id']}/{key}.mp4"
        out = os.path.join(DATA_DIR, rel)
        m = {"sample_id": r["sample_id"], "key": key, "video_id": r["video_id"], "window": [s, e]}
        cover = sorted([(cp, a, b) for cp, (a, b) in vc if a <= s + EPS and b >= e - EPS], key=lambda x: x[2] - x[1])
        full = os.path.join(DATA_DIR, "ego4d_full", f"{r['video_id']}.mp4")
        parts, job = None, None
        if cover:
            cp, a, b = cover[0]
            parts = [(os.path.join(DATA_DIR, cp), max(0.0, s - a), min(b - a, e - a))]
            job = ("clip", parts)
            m.update(status="cut", method="single_clip", from_clips=[cp])
        elif os.path.exists(full):
            sizes = [vinfo(os.path.join(DATA_DIR, cp)) for cp, _ in vc]
            size = max(sizes, key=lambda wh: wh[0] * wh[1])
            fw, fh = vinfo(full)
            crop_left = fw / fh > 1.5 * size[0] / size[1]            # side-by-side source → dataset clips use the left half
            job = ("full", (full, s, e, crop_left, size))
            m.update(status="cut", method="full_source", source_video=os.path.relpath(full, DATA_DIR), crop_left_half=crop_left,
                     output_size=list(size))
        else:                                  # greedy chain of overlapping clips
            t, chain = s, []
            while t < e - EPS:
                nxt = max(((cp, a, b) for cp, (a, b) in vc if a <= t + EPS and b > t + EPS), key=lambda x: x[2], default=None)
                if nxt is None:
                    break
                chain.append((os.path.join(DATA_DIR, nxt[0]), t - nxt[1], min(nxt[2], e) - nxt[1]))
                t = min(nxt[2], e)
            if t >= e - EPS and chain:
                size = max((vinfo(p[0]) for p in chain), key=lambda wh: wh[0] * wh[1])
                job = ("stitch", (chain, size))
                m.update(status="cut", method=f"stitched_{len(chain)}_clips", from_clips=[os.path.relpath(p[0], DATA_DIR) for p in chain])
            else:
                covered = max((min(b, e) - max(a, s) for _, (a, b) in vc), default=0)
                m.update(status="skipped", reason="window not covered by existing egoOmni clips and full Ego4D video unavailable",
                         missing_sec=round((e - s) - max(covered, 0), 2))
        parts = job
        if parts:
            if not os.path.exists(out):
                os.makedirs(os.path.dirname(out), exist_ok=True)
                tmp = out + ".tmp.mp4"
                kind, arg = job
                if kind == "clip":
                    cut_one(*arg[0], tmp)
                elif kind == "full":
                    cut_full(*arg, tmp)
                else:
                    cut_stitched(arg[0], tmp, arg[1])
                os.replace(tmp, out)
            if job[0] == "full":                  # pixel check against a dataset clip overlapping the window
                ov = [(cp, a, b) for cp, (a, b) in vc if min(b, e) - max(a, s) > 2]
                if ov:
                    cp, a, b = ov[0]
                    t_src = (max(a, s) + min(b, e)) / 2
                    d = float(abs(gray(out, t_src - s) - gray(os.path.join(DATA_DIR, cp), t_src - a)).mean())
                    m["pixel_check_vs_clip"] = {"clip": cp, "t_src": round(t_src, 2), "mean_abs_diff_0_255": round(d, 2)}
            req = tuple(sorted(q["loop_annotation"].get("required_modalities") or []))
            items.append({
                "sample_id": r["sample_id"], "qa_id": None, "video_id": r["video_id"], "clip_path": rel,
                "clip_duration_sec": round(e - s, 3), "source_clip_interval": {"start_sec": s, "end_sec": e},
                "timestamp_reference": "clip_seconds", "split": r["split"], "source_kind": "restored_single",
                "category": r["category"], "subcategory": q.get("routed_subcategory"), "benchmark_track": None,
                "original_question_format": "open", "options": None, "correct_options": [],
                "original_question": r["original_question"], "original_answer": r["original_answer"],
                "minimum_modalities": MODS.get(req), "required_modalities": list(req), "modalities": None,
                "round_count": 1, "depends_on_earlier_rounds": bool((q.get("dialogue") or {}).get("depends_on_round_ids")),
                "reasoning_level": (r.get("taxonomy_v2") or {}).get("reasoning_level"),
                "restoration_previous_status": r["restoration"].get("previous_status"),
                "quality_verified": r["restoration"].get("quality_verified"),
                "qa": [{"turn_index": 1, "question": r["original_question"], "answer": r["original_answer"]}]})
        manifest.append(m)
    json.dump(items, open(os.path.join(DATA_DIR, "test", "qa_restored_v3.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(manifest, open(os.path.join(DATA_DIR, "test", "restored_v3_manifest.json"), "w"), ensure_ascii=False, indent=1)
    st = defaultdict(int)
    for m in manifest:
        st[m.get("method") or m["status"]] += 1
    print(f"restored rows={len(restored)} → evaluable={len(items)}  {dict(st)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test/qa_v2_restored.jsonl")
