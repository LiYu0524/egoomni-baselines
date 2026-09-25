#!/usr/bin/env python3
"""MiniCPM-o-2_6 adapter — model settings copied from EgoSound's scripts/minicpm_av_infer.py (eval_omni):
get_video_chunk_new = one "<unit>" per second (every 2 s for clips > 240 s) holding one frame (moviepy get_frame) and
that second of 16 kHz mono audio; omni system prompt (mode='omni', language='en'); omni_input, use_tts_template,
max_slice_nums=1, use_image_id=False, sdpa, bf16. Decoding: EgoSound's script samples (temperature 0.5); for EgoToM /
EgoTempo we decode greedily (sampling=False) so the multiple-choice / short answers are deterministic — set
MINICPM_SAMPLING=1 to reproduce the EgoSound setting. A clip without an audio track gets silence of the same length.
A benchmark system prompt (EgoToM) is placed before the question inside the user turn (the omni system prompt stays).
usage: infer_minicpmo.py MODEL_DIR REQUESTS OUT_DIR SHARD NUM_SHARDS
"""
import math, os, sys, tempfile
import numpy as np
import torch
from PIL import Image

MODEL_DIR, REQ, OUT, SHARD, NSH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_infer as C  # noqa: E402
import librosa  # noqa: E402
from moviepy.video.io.VideoFileClip import VideoFileClip  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

SAMPLING = os.environ.get("MINICPM_SAMPLING") == "1"
model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model.init_tts()


def get_video_chunk_new(video_path):
    video = VideoFileClip(video_path)
    if video.audio is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            video.audio.write_audiofile(tmp.name, codec="pcm_s16le", fps=16000, verbose=False, logger=None)
            audio_np, sr = librosa.load(tmp.name, sr=16000, mono=True)
    else:
        sr = 16000
        audio_np = np.zeros(int(math.ceil(video.duration) * sr) + sr, dtype=np.float32)
    sample_interval = 2 if video.duration > 240 else 1
    contents = []
    for i in range(math.ceil(video.duration / sample_interval)):
        t = min((i + 1) * sample_interval, video.duration - 0.001)
        image = Image.fromarray(video.get_frame(t).astype(np.uint8))
        audio = audio_np[int(sr * (t - sample_interval)):int(sr * t)]
        contents.extend(["<unit>", image, audio])
    has_audio = video.audio is not None
    video.close()
    return contents, has_audio


def infer(r):
    contents, has_audio = get_video_chunk_new(C.media(r["video"]))
    question = f"{r['system']}\n\n{r['user']}" if r.get("system") else r["user"]
    msgs = [model.get_sys_prompt(mode="omni", language="en"), {"role": "user", "content": contents + [question]}]
    kw = dict(sampling=True, temperature=0.5) if SAMPLING else dict(sampling=False)
    res = model.chat(msgs=msgs, tokenizer=tokenizer, max_new_tokens=r["max_new_tokens"], omni_input=True,
                     use_tts_template=True, generate_audio=False, max_slice_nums=1, use_image_id=False, return_dict=True, **kw)
    return res.text.strip(), {"units": len(contents) // 3, "audio": has_audio}


C.run("minicpmo", REQ, OUT, SHARD, NSH, infer)
