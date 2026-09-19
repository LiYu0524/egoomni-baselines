#!/usr/bin/env python
"""Minimal VideoLLaMA2.1-7B-AV audio-visual inference on one video (README 'Audio/Video-Audio Inference').
usage: python videollama2_infer.py --model models/VideoLLaMA2.1-7B-AV --video clip.mp4 [--modal av|v|a] [--prompt ...]
"""
import argparse, os, sys, time
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")  # env ffmpeg for audio extraction
REPO = os.environ.get("VIDEOLLAMA2_REPO", "/ai4good1-shared/liyu/egoOmni_baselines/repos/VideoLLaMA2")
sys.path.insert(0, REPO)
import torch
from videollama2 import model_init, mm_infer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--video", required=True)
ap.add_argument("--modal", choices=["av", "v", "a"], default="av")
ap.add_argument("--prompt", default="Describe the video in detail, including what is said and heard in the audio.")
a = ap.parse_args(); t0 = time.time()

model, processor, tokenizer = model_init(a.model)
if a.modal == "a": model.model.vision_tower = None
elif a.modal == "v": model.model.audio_tower = None
print(f"[{time.time()-t0:.0f}s] model loaded: {torch.cuda.memory_allocated()/2**30:.1f} GiB on GPU")
if a.modal == "a":
    tensor = processor["audio"](a.video)
else:
    tensor = processor["video"](a.video, va=(a.modal == "av"))
out = mm_infer(tensor, a.prompt, model=model, tokenizer=tokenizer, modal=("audio" if a.modal == "a" else "video"), do_sample=False)
print(f"[{time.time()-t0:.0f}s] ===== OUTPUT =====\n{out}")
