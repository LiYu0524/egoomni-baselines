#!/usr/bin/env python3
"""Frame-based HF inference for Qwen2-VL-7B-Instruct and LLaVA-OneVision-Qwen2-7B (transformers >= 4.45).

usage: infer_hf.py {qwen2vl|llavaov} MODEL_DIR REQUESTS_JSONL OUT_DIR SHARD NUM_SHARDS
env: MEDIA_ROOT (local media root filled by stage_media.py), LIMIT (smoke test)
Frames: the request's frame_mode / nframes — 'uniform_end' is EgoToM's official select_video_frames('uniformN')
(end-aligned), 'uniform' is np.linspace over the clip. Video-only (these models have no audio input).
Greedy decoding. One JSON line per request in OUT_DIR/shard_<k>.jsonl; finished ids are skipped on restart.
"""
import json, os, sys, time, traceback
import numpy as np
import torch
from decord import VideoReader, cpu

KIND, MODEL_DIR, REQ, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SHARD, NSH = int(sys.argv[5]), int(sys.argv[6])
MEDIA = os.environ.get("MEDIA_ROOT", "/tmp/egobench_media")
LIMIT = int(os.environ.get("LIMIT", "0"))


def select_frames(total, n, mode):
    if total <= 0:
        raise ValueError("empty video")
    if mode == "uniform_end":                       # EgoToM utils.select_video_frames('uniform{n}')
        step = total // n
        if step == 0:
            return np.arange(total)
        return np.flip(np.arange(total - 1, -1, -step)[:n])
    return np.unique(np.linspace(0, total - 1, n).round().astype(int))


def load_frames(path, n, mode):
    try:
        vr = VideoReader(path, ctx=cpu(0), num_threads=2)
        idx = select_frames(len(vr), n, mode)
        frames = vr.get_batch(list(idx)).asnumpy()
    except Exception:                               # decord cannot decode a few files (e.g. EgoSound 01306): OpenCV fallback
        import cv2
        cap = cv2.VideoCapture(path)
        allf = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            allf.append(fr[:, :, ::-1])
        cap.release()
        idx = select_frames(len(allf), n, mode)
        frames = np.stack([allf[i] for i in idx])
    if KIND == "qwen2vl" and len(frames) % 2:       # Qwen2-VL merges frame pairs; pad by repeating the last frame
        frames = np.concatenate([frames, frames[-1:]], 0)
    return frames


def build():
    if KIND == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16,
                                                                attn_implementation="sdpa")
    elif KIND == "llavaov":
        from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16,
                                                                       attn_implementation="sdpa")
    else:
        raise SystemExit(f"unknown model kind {KIND}")
    return model.to("cuda").eval(), AutoProcessor.from_pretrained(MODEL_DIR)   # no device_map: env has no accelerate


def main():
    rows = [json.loads(l) for l in open(REQ)][SHARD::NSH]
    if LIMIT:
        rows = rows[:LIMIT]
    os.makedirs(OUT, exist_ok=True)
    outf = os.path.join(OUT, f"shard_{SHARD}.jsonl")
    done = set()                                    # answered in ANY shard file (a re-run may use another sharding)
    import glob
    for f in glob.glob(os.path.join(OUT, "shard_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if "pred" in r:
                done.add(r["id"])
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{KIND}] shard {SHARD}/{NSH}: {len(rows)} rows, {len(todo)} to do", flush=True)
    if not todo:
        return
    model, proc = build()
    t0 = time.time()
    with open(outf, "a") as fo:
        for i, r in enumerate(todo):
            rec = {"id": r["id"]}
            try:
                frames = load_frames(os.path.join(MEDIA, r["video"]), r["nframes"], r["frame_mode"])
                msgs = []
                if r.get("system"):
                    msgs.append({"role": "system", "content": [{"type": "text", "text": r["system"]}]})
                msgs.append({"role": "user", "content": [{"type": "video"}, {"type": "text", "text": r["user"]}]})
                prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
                kw = {"do_sample_frames": False} if KIND == "qwen2vl" else {}
                try:
                    inputs = proc(text=[prompt], videos=[frames], return_tensors="pt", **kw)
                except TypeError:
                    inputs = proc(text=[prompt], videos=[frames], return_tensors="pt")
                inputs = inputs.to(model.device, torch.bfloat16) if hasattr(inputs, "to") else inputs
                with torch.inference_mode():
                    out = model.generate(**inputs, do_sample=False, max_new_tokens=r["max_new_tokens"])
                pred = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
                rec.update(pred=pred, nframes=int(len(frames)), in_tokens=int(inputs["input_ids"].shape[1]))
            except Exception as e:
                rec.update(error=f"{type(e).__name__}: {str(e)[:300]}")
                traceback.print_exc()
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"[{KIND}] shard {SHARD}: {i + 1}/{len(todo)} {el / (i + 1):.2f}s/item", flush=True)
    print(f"[{KIND}] shard {SHARD}: DONE", flush=True)


if __name__ == "__main__":
    main()
