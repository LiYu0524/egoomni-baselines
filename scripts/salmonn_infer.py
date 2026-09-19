#!/usr/bin/env python
"""Minimal video-SALMONN 2+ inference on one video (merged model dir). Mirrors repo inference.py with
the paper's eval settings (--max_frames 768 --max_pixels 61250 --interval 0.1).
usage: python salmonn_infer.py --model models/video-SALMONN-2_plus_7B_merged --video clip.mp4 [--prompt ...] [--no_audio]
"""
import argparse, os, sys, time
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")  # env ffmpeg for audio extraction
import torch
REPO = os.environ.get("SALMONN_REPO", "/ai4good1-shared/liyu/egoOmni_baselines/repos/video-SALMONN-2/video_SALMONN2_plus")
ORIG_CWD = os.getcwd()
sys.path.insert(0, REPO); os.chdir(REPO)   # qwenvl code expects the repo as cwd
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import make_supervised_data_module
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from qwenvl.train.argument import DataArguments
from transformers import AutoTokenizer, WhisperFeatureExtractor
from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from qwenvl.model import modeling_qwen2_5_vl as M
M.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb; M.Qwen2RMSNorm = LigerRMSNorm; M.Qwen2MLP = LigerSwiGLUMLP

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--video", required=True)
ap.add_argument("--prompt", default="Describe the video in detail, including what is said and heard in the audio.")
ap.add_argument("--no_audio", action="store_true"); ap.add_argument("--max_frames", type=int, default=768)
ap.add_argument("--max_pixels", type=int, default=61250); ap.add_argument("--interval", type=float, default=0.1)
ap.add_argument("--max_new_tokens", type=int, default=512)
ap.add_argument("--device_map", default="cuda", help="cuda = whole model on one GPU (7B); auto = shard across visible GPUs (72B, ~145 GB bf16)")
a = ap.parse_args(); t0 = time.time()
a.model, a.video = (os.path.join(ORIG_CWD, p) for p in (a.model, a.video))   # resolve against the caller cwd, not the repo

d = DataArguments()
d.video_max_frames, d.video_min_frames, d.base_interval = a.max_frames, 16, a.interval
d.max_pixels = d.video_max_frame_pixels = a.max_pixels
d.run_test = True; d.model_type = "qwen2.5vl"
d.image_processor = Qwen2VLImageProcessorFast.from_pretrained(a.model)
d.audio_processor = WhisperFeatureExtractor(feature_size=d.feature_size, sampling_rate=d.sampling_rate, hop_length=d.hop_length, chunk_length=d.chunk_length)
tok = AutoTokenizer.from_pretrained(a.model, model_max_length=131072, padding_side="right", use_fast=False)
if a.device_map == "cuda":
    model = video_SALMONN2_plus.from_pretrained(a.model, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="cpu").cuda().eval()
else:
    model = video_SALMONN2_plus.from_pretrained(a.model, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map=a.device_map).eval()
print(f"[{time.time()-t0:.0f}s] model loaded: " + ", ".join(f"gpu{i}={torch.cuda.memory_allocated(i)/2**30:.1f}GiB" for i in range(torch.cuda.device_count()) if torch.cuda.memory_allocated(i)))

ds = make_supervised_data_module(tokenizer=tok, data_args=d)["train_dataset"]
item = {"video": a.video, "use_audio": not a.no_audio,
        "conversations": [{"from": "human", "value": "<video>\n" + a.prompt}, {"from": "gpt", "value": ""}]}
inp = ds._get_item(item)
for k in ("video", "image", "prompt", "ref", "audio", "use_audio", "should_use"): inp.pop(k, None)
inp = {k: v.cuda() for k, v in inp.items() if isinstance(v, torch.Tensor)}
print(f"[{time.time()-t0:.0f}s] input tokens: {inp['input_ids'].shape[1]}")
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
print(f"[{time.time()-t0:.0f}s] ===== OUTPUT =====")
print(tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))
