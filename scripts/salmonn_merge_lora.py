#!/usr/bin/env python
"""Build a ready-to-load video-SALMONN 2+ model from the released LoRA checkpoint.

Reproduces the lora_ckpt branch of train_qwen.py (run_test) exactly once and saves the merged result:
  PeftModel.from_pretrained(base, lora) -> merge_and_unload -> save_pretrained
Afterwards inference is a plain video_SALMONN2_plus.from_pretrained(<out>) (see salmonn_infer.py),
or scripts/test.sh with --model <out> --model_base <out> and no --lora_ckpt.

The base MUST be the audio-aligned checkpoint (tsinghua-ee/video_SALMONN2plus_{7B,72B}_audioAlign):
audio.q_tokens is a bare nn.Parameter, so peft's modules_to_save could not store it in the LoRA and it was
frozen during LoRA SFT -> the trained queries only exist in the audio-aligned full checkpoint.
(--whisper is only for reproducing gen_audio_model.py on a plain Qwen2.5-VL base, which yields RANDOM q_tokens.)

usage:
  python salmonn_merge_lora.py --base models/video_SALMONN2plus_7B_audioAlign \
                               --lora models/video-SALMONN-2_plus_7B --out models/video-SALMONN-2_plus_7B_merged
"""
import argparse, os, shutil, sys, time
import torch

REPO = os.environ.get("SALMONN_REPO", "/ai4good1-shared/liyu/egoOmni_baselines/repos/video-SALMONN-2/video_SALMONN2_plus")
sys.path.insert(0, REPO)
from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus  # noqa: E402
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq  # noqa: E402
from peft import PeftModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="tsinghua-ee/video_SALMONN2plus_{7B,72B}_audioAlign dir")
ap.add_argument("--whisper", default=None, help="(gen_audio_model.py reproduction only) openai/whisper-large-v3 dir; NOT for eval")
ap.add_argument("--lora", required=True, help="tsinghua-ee/video-SALMONN-2_plus_{7B,72B} dir")
ap.add_argument("--out", required=True)
args = ap.parse_args()
t0 = time.time()

import json
base_keys = json.load(open(os.path.join(args.base, "model.safetensors.index.json")))["weight_map"]
base_has_audio = "audio.q_tokens" in base_keys
if not base_has_audio and args.whisper is None:
    sys.exit(f"{args.base} has no trained audio.q_tokens -> use the *_audioAlign checkpoint as --base (or pass --whisper to reproduce gen_audio_model.py, which is NOT the eval recipe)")

tokenizer = AutoTokenizer.from_pretrained(args.base, model_max_length=131072, padding_side="right", use_fast=False)
if not base_has_audio:
    tokenizer.add_tokens(["<|audio_pad|>"])
assert tokenizer.convert_tokens_to_ids("<|audio_pad|>") == 151665, "audio_token_id mismatch with config default"

model = video_SALMONN2_plus.from_pretrained(args.base, attn_implementation="flash_attention_2",
                                            torch_dtype=torch.bfloat16, device_map="cpu")
print(f"[{time.time()-t0:.0f}s] base loaded (audio weights in base: {base_has_audio})")
if not base_has_audio:   # --- gen_audio_model.py reproduction ---
    whisper = AutoModelForSpeechSeq2Seq.from_pretrained(args.whisper, torch_dtype=torch.bfloat16)
    enc_sd = whisper.model.encoder.state_dict()
    copied, skipped = 0, []
    for k, v in model.audio.named_parameters():
        if k in enc_sd and v.shape == enc_sd[k].shape:
            v.data = enc_sd[k].data; copied += 1
        else:
            skipped.append(k)
    model.audio.q_tokens.data.normal_(mean=0.0, std=0.02)
    print(f"[{time.time()-t0:.0f}s] whisper encoder params copied: {copied}; not in whisper: {len(skipped)}  (q_tokens are RANDOM)")
    del whisper, enc_sd

# --- train_qwen.py lora_ckpt branch ---
audio_layers = model.audio.layers
del model.audio.layers                      # keep LoRA off the whisper attention (q/k/v_proj would match)
model = PeftModel.from_pretrained(model, args.lora)
model.model.audio.layers = audio_layers
model = model.merge_and_unload()
print(f"[{time.time()-t0:.0f}s] LoRA merged")

os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(args.out)
for f in ("chat_template.json", "preprocessor_config.json"):
    shutil.copy(os.path.join(args.base, f), os.path.join(args.out, f))
open(os.path.join(args.out, "PROVENANCE.txt"), "w").write(
    f"base={args.base}\nbase_has_trained_audio={base_has_audio}\nwhisper={args.whisper}\nlora={args.lora}\nbuilt={time.strftime('%F %T')}\nby=salmonn_merge_lora.py (train_qwen.py lora_ckpt merge)\n")
print(f"[{time.time()-t0:.0f}s] saved -> {args.out}")
