#!/usr/bin/env python
"""Cross-check the LoRA+audioAlign recipe against the official merged release (tsinghua-ee/video-SALMONN2_plus_7B_full).
Only shard 1 of _full is needed (models/video-SALMONN2_plus_7B_full). Expected:
  q_tokens: audioAlign == full           (frozen during LoRA SFT; NOT in the adapter)
  qformer/audio_proj: adapter modules_to_save == full, != audioAlign   (trained during SFT)
  whisper encoder: audioAlign == full == whisper-large-v3            (frozen throughout)
  LLM q_proj: audioAlign + B@A*(alpha/r) ≈ full  (bf16 rounding), audioAlign == Qwen2.5-VL (LLM untouched by alignment)
"""
import json, torch
from safetensors import safe_open
ROOT = "/ai4good1-shared/liyu/egoOmni_baselines/models"
FULL, AL, QW = f"{ROOT}/.validate_full_7B", f"{ROOT}/video_SALMONN2plus_7B_audioAlign", f"{ROOT}/Qwen2.5-VL-7B-Instruct"
ad = safe_open(f"{ROOT}/video-SALMONN-2_plus_7B/adapter_model.safetensors", "pt"); akeys = list(ad.keys())
def T(d, k):
    wm = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]; return safe_open(f"{d}/{wm[k]}", "pt").get_tensor(k).float()
def A(suffix): return ad.get_tensor([x for x in akeys if x.endswith(suffix)][0]).float()
def cmp(a, b): return "EQUAL" if torch.equal(a, b) else f"DIFF (rel max err {((a-b).abs().max()/(b.abs().max()+1e-9)).item():.3g})"
k = "audio.q_tokens";                                             print(f"{'q_tokens: audioAlign vs full':52s}", cmp(T(AL, k), T(FULL, k)))
k = "audio.qformer.bert.encoder.layer.0.attention.self.query.weight"
print(f"{'qformer.l0.query: audioAlign vs full':52s}", cmp(T(AL, k), T(FULL, k)))
print(f"{'qformer.l0.query: adapter(modules_to_save) vs full':52s}", cmp(A(k), T(FULL, k)))
k = "audio.audio_proj.weight";                                   print(f"{'audio_proj: adapter(modules_to_save) vs full':52s}", cmp(A(k), T(FULL, k)))
k = "visual.merger.mlp.0.weight";                                print(f"{'visual.merger: adapter(modules_to_save) vs full':52s}", cmp(A(k), T(FULL, k)))
k = "audio.layers.0.self_attn.q_proj.weight"
w = safe_open(f"{ROOT}/whisper-large-v3/model.safetensors", "pt").get_tensor("model.encoder.layers.0.self_attn.q_proj.weight").float()
print(f"{'whisper.l0.q_proj: audioAlign vs full':52s}", cmp(T(AL, k), T(FULL, k)), "| audioAlign vs whisper-large-v3:", cmp(T(AL, k), w))
k = "model.layers.0.self_attn.q_proj.weight"
b, f, q = T(AL, k), T(FULL, k), T(QW, k)
la = [x for x in akeys if "model.layers.0.self_attn.q_proj" in x and "lora_A" in x][0]
merged = (b + (ad.get_tensor(la.replace("lora_A", "lora_B")).float() @ ad.get_tensor(la).float()) * (256 / 128)).bfloat16().float()
print(f"{'llm.l0.q_proj: audioAlign vs Qwen2.5-VL-7B':52s}", cmp(b, q))
print(f"{'llm.l0.q_proj: audioAlign vs full (no LoRA)':52s}", cmp(b, f))
print(f"{'llm.l0.q_proj: audioAlign + LoRA(alpha/r=2) vs full':52s}", cmp(merged, f))
