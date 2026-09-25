"""Shared driver for the per-model adapters: shard selection, resume, one JSON line per request.

    run(tag, requests_path, out_dir, shard, nshards, infer)   where infer(row, media_path_fn) -> (pred, extra_dict)
"""
import json, os, time, traceback

MEDIA = os.environ.get("MEDIA_ROOT", "/egobench_media")
LIMIT = int(os.environ.get("LIMIT", "0"))


def media(rel):
    return os.path.join(MEDIA, rel) if rel else None


def run(tag, req, out, shard, nsh, infer):
    rows = [json.loads(l) for l in open(req)][shard::nsh]
    if LIMIT:
        rows = rows[:LIMIT]
    os.makedirs(out, exist_ok=True)
    outf = os.path.join(out, f"shard_{shard}.jsonl")
    done = set()                                    # answered in ANY shard file (a re-run may use another sharding)
    import glob
    for f in glob.glob(os.path.join(out, "shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if "pred" in r:
                done.add(r["id"])
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{tag}] shard {shard}/{nsh}: {len(rows)} rows, {len(todo)} to do", flush=True)
    t0 = time.time()
    with open(outf, "a") as fo:
        for i, r in enumerate(todo):
            rec = {"id": r["id"]}
            try:
                pred, extra = infer(r)
                rec.update(pred=pred, **(extra or {}))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                traceback.print_exc()
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            if (i + 1) % 25 == 0:
                print(f"[{tag}] shard {shard}: {i + 1}/{len(todo)} {(time.time() - t0) / (i + 1):.2f}s/item", flush=True)
    print(f"[{tag}] shard {shard}: DONE", flush=True)


def audio_16k(path):
    """Mono 16 kHz float32 audio of a media file via PyAV; None if it has no audio stream."""
    import av, numpy as np
    with av.open(path) as c:
        if not c.streams.audio:
            return None
        rs = av.AudioResampler(format="flt", layout="mono", rate=16000)
        chunks = []
        for fr in c.decode(audio=0):
            for o in rs.resample(fr):
                chunks.append(o.to_ndarray().reshape(-1))
        for o in rs.resample(None):
            chunks.append(o.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype("float32") if chunks else None
