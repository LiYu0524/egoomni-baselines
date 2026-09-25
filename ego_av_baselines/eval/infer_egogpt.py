#!/usr/bin/env python3
"""EgoGPT-7b-EgoIT-EgoLife adapter — model settings copied from EgoSound's scripts/egogpt_av_inference.py:
video at 1 fps capped to 16 uniform frames (decord), audio -> whisper.pad_or_trim (first 30 s) -> 128-bin log-mel,
qwen_1_5 conversation "<image>\\n<speech>\\n\\n{query}", greedy, fp16.
Audio source: the benchmark's .wav when the request names one (EgoSound, as in the official script), else the clip's own
audio track (PyAV); silent input (zeros(3000, 128)) when there is none, exactly the script's no-audio branch.
A benchmark system prompt (EgoToM) replaces the template's default system message.
The config refers to './large-v3.pt' and 'google/siglip-so400m-patch14-384'; both resolve from EGOGPT_CWD (symlinks).
usage: infer_egogpt.py MODEL_DIR REQUESTS OUT_DIR SHARD NUM_SHARDS
"""
import copy, os, re, sys, warnings
import numpy as np
import torch

MODEL_DIR, REQ, OUT, SHARD, NSH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_infer as C  # noqa: E402

os.chdir(os.environ.get("EGOGPT_CWD", os.path.join(os.path.dirname(MODEL_DIR), "egogpt_cwd")))
warnings.filterwarnings("ignore")
import soundfile as sf  # noqa: E402
import whisper  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402
from scipy.signal import resample  # noqa: E402
import torch.distributed as dist  # noqa: E402
from egogpt.constants import IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX  # noqa: E402
from egogpt.conversation import conv_templates  # noqa: E402
from egogpt.model.builder import load_pretrained_model  # noqa: E402

os.environ["MASTER_ADDR"] = "localhost"
os.environ.setdefault("MASTER_PORT", str(12358 + SHARD))
dist.init_process_group("gloo", rank=0, world_size=1)          # as in the official script
tokenizer, model, _ = load_pretrained_model(MODEL_DIR, device_map="cuda")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()


def speech_features(r):
    wav = C.media(r.get("audio"))
    if wav and os.path.exists(wav):
        speech, sr = sf.read(wav)
        if sr != 16000:
            speech = resample(speech, int(len(speech) * 16000 / sr))
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
    else:
        speech = C.audio_16k(C.media(r["video"]))
    if speech is None or len(speech) == 0:
        return torch.zeros(3000, 128), torch.LongTensor([3000]), False
    speech = whisper.pad_or_trim(speech.astype(np.float32))
    mel = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
    return mel, torch.LongTensor([mel.shape[0]]), True


def load_video(path, max_frames_num=16, fps=1):
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    step = max(1, round(vr.get_avg_fps() / fps))
    idx = list(range(0, total, step))
    if max_frames_num > 0 and len(idx) > max_frames_num:
        idx = np.linspace(0, total - 1, max_frames_num, dtype=int).tolist()
    return vr.get_batch(idx).asnumpy()


def infer(r):
    conv = copy.deepcopy(conv_templates["qwen_1_5"])
    if r.get("system"):
        conv.system = f"<|im_start|>system\n{r['system']}"
    conv.append_message(conv.roles[0], f"<image>\n<speech>\n\n{r['user']}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    video = load_video(C.media(r["video"]))
    speech, speech_lengths, has_audio = speech_features(r)
    speech = torch.stack([speech]).to("cuda").half()
    pv = model.get_vision_tower().image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
    ids = []
    for part in [p for p in re.split("(<image>|<speech>)", prompt) if p]:
        ids += [IMAGE_TOKEN_INDEX] if part == "<image>" else [SPEECH_TOKEN_INDEX] if part == "<speech>" else tokenizer(part).input_ids
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to("cuda")
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), images=[pv.half()], image_sizes=[video[0].size],
                             speech=speech, speech_lengths=speech_lengths, do_sample=False, max_new_tokens=r["max_new_tokens"],
                             modalities=["video"], eos_token_id=tokenizer.eos_token_id)
    pred = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    return pred, {"nframes": int(len(video)), "audio": has_audio}


C.run("egogpt", REQ, OUT, SHARD, NSH, infer)
