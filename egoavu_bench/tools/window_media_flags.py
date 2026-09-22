#!/usr/bin/env python3
"""Measure what each bench window actually contains, from the exact media the models saw (proxy + #t=a,b window):
mean luma of 8 frames evenly spaced in the window and audio RMS over the whole window.
-> data/window_media_flags.json {bench_idx: {...}}. A frame counts as black if its mean luma < 2 (0-255)."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

BI = "/ai4good1-shared/liyu/egoavu/bench_infer"

def split(p):
    path, frag = p.split("#t=")
    a, b = (float(x) for x in frag.split(","))
    return path, a, b

def measure(row):
    import av, soundfile as sf
    vpath, a, b = split(row["videos"][0])
    times = [a + (b - a) * (i + 0.5) / 8 for i in range(8)]
    lumas = []
    with av.open(vpath) as c:
        vs = c.streams.video[0]; tb = vs.time_base
        for t in times:
            c.seek(int(t / tb), stream=vs, backward=True, any_frame=False)
            y = None
            for fr in c.decode(vs):
                y = fr
                if fr.pts is not None and fr.pts * tb >= t - 0.26:
                    break
            lumas.append(round(float(y.to_ndarray(format="gray").mean()), 2) if y is not None else None)
    rms = None
    if row["audios"]:
        apath, aa, ab = split(row["audios"][0])
        sr = sf.info(apath).samplerate
        x, _ = sf.read(apath, start=int(aa * sr), stop=int(ab * sr), dtype="float32", always_2d=False)
        rms = round(float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0, 6)
    black = sum(1 for l in lumas if l is not None and l < 2.0)
    return row["bench_idx"], {"frame_luma_8": lumas, "black_frames_of_8": black,
                              "video_all_black": black == 8, "audio_rms": rms,
                              "audio_silent": (rms is not None and rms < 1e-4)}

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(f"{BI}/data/egoavu_bench_eval.jsonl")]
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 48) as ex:
        res = dict(ex.map(measure, rows, chunksize=8))
    json.dump({str(k): res[k] for k in sorted(res)}, open(f"{BI}/data/window_media_flags.json", "w"))
    n_black = sum(v["video_all_black"] for v in res.values()); n_part = sum(0 < v["black_frames_of_8"] < 8 for v in res.values())
    n_sil = sum(v["audio_silent"] for v in res.values()); n_both = sum(v["video_all_black"] and v["audio_silent"] for v in res.values())
    print(json.dumps({"windows": len(res), "video_all_black": n_black, "video_partly_black": n_part, "audio_silent": n_sil,
                      "all_black_and_silent": n_both, "measure_failures": sum(None in v["frame_luma_8"] for v in res.values())}))
