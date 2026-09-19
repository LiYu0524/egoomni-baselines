#!/usr/bin/env python
"""Rebuild every tensor of _full shard 1 from (audioAlign base, LoRA adapter) and report all mismatches."""
import json, torch
from safetensors import safe_open
ROOT = "/ai4good1-shared/liyu/egoOmni_baselines/models"
FULL, AL = f"{ROOT}/.validate_full_7B", f"{ROOT}/video_SALMONN2plus_7B_audioAlign"
ad = safe_open(f"{ROOT}/video-SALMONN-2_plus_7B/adapter_model.safetensors", "pt"); akeys = set(ad.keys())
wm_al = json.load(open(f"{AL}/model.safetensors.index.json"))["weight_map"]
opened = {}
def base(k):
    f = wm_al[k]; opened.setdefault(f, safe_open(f"{AL}/{f}", "pt")); return opened[f].get_tensor(k)
full = safe_open(f"{FULL}/model-00001-of-00004.safetensors", "pt")
n = mism = 0; kinds = {"lora": 0, "m2s": 0, "base": 0}
for k in full.keys():
    f = full.get_tensor(k); n += 1
    stem = k[:-len(".weight")] if k.endswith(".weight") else None
    la = f"base_model.model.{stem}.lora_A.weight" if stem else None
    m2s = f"base_model.model.{k}"
    if la and la in akeys:
        r = (base(k).float() + ad.get_tensor(la.replace("lora_A", "lora_B")).float() @ ad.get_tensor(la).float() * 2.0).to(f.dtype); kinds["lora"] += 1
    elif m2s in akeys:
        r = ad.get_tensor(m2s).to(f.dtype); kinds["m2s"] += 1
    else:
        r = base(k); kinds["base"] += 1
    if r.shape != f.shape or not torch.equal(r, f):
        mism += 1; print("MISMATCH", k, tuple(f.shape), f"rel max err {((r.float()-f.float()).abs().max()/(f.float().abs().max()+1e-9)).item():.3g}")
print(f"tensors checked: {n}  (lora-merged {kinds['lora']}, modules_to_save {kinds['m2s']}, base {kinds['base']})  mismatches: {mism}")
