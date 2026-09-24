**Table 1 — original protocol** (LLaMAFactory, EgoAVU media recipe: 2 fps ≤ 64 frames, 200,704 px)

| model | EAB SSA S | EAB AVSN S | EAB AVSN M | EAB TR | EAB AVH | egoOmni gold | egoOmni self | EgoCross | EgoSchema | ETQ direct EM | ETQ direct judge | ETQ indirect EM | ETQ indirect judge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base (untuned Qwen2.5-Omni-7B) | 1.50 | 1.92 | 10.47 | 44.6 | 23.6 | 36.80 | 35.01 | 45.25 | 65.2 | 29.96 | 34.52 | 34.62 | 38.17 |
| r100k (ours) | 2.65 | 2.01 | 20.59 | 44.2 | 98.4 | 28.69 | 26.60 | 43.89 | 58.0 | 22.16 | 29.50 | 27.86 | 33.28 |
| r20k-8gpu (ours) | 2.65 | 2.05 | 19.33 | 42.0 | 97.5 | 30.12 | 28.77 | 45.66 | 64.0 | 25.06 | 30.88 | 29.77 | 34.17 |
| r20k-32gpu (ours) | 2.62 | 2.06 | 19.57 | 47.2 | 97.1 | 29.56 | 27.94 | 45.14 | 60.4 | 24.50 | 30.45 | 29.35 | 34.12 |
| ckpt_sft (colleague LoRA) | 1.53 | 1.68 | 4.39 | 43.6 | 13.0 | 52.30 | 50.41 | 44.41 | 56.4 | 25.65 | 33.31 | 31.02 | 36.81 |
| ckpt_epoch2 = ckpt_lora_epoch2 (colleague LoRA) | 1.56 | 1.67 | 5.11 | 45.0 | 36.1 | 45.58 | 43.68 | 43.57 | 54.2 | 24.52 | 33.71 | 31.04 | 38.65 |
| ckpt_fft_epoch2 (colleague full FT) | 1.56 | 1.69 | 5.46 | 44.8 | 22.5 | 48.36 | 46.70 | 41.07 | 57.6 | 23.10 | 32.62 | 29.07 | 36.41 |

**Table 2 — colleague's EgoToM-kit protocol** (native HF + qwen_omni_utils, 2 fps ≤ 360 frames, 156,800 px, audio-in-video; cell = kit (Δ vs Table 1))

| model | EAB SSA S | EAB AVSN S | EAB AVSN M | EAB TR | EAB AVH | egoOmni gold | egoOmni self | EgoCross | EgoSchema | ETQ direct EM | ETQ direct judge | ETQ indirect EM | ETQ indirect judge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base (untuned Qwen2.5-Omni-7B) | 1.45 (-0.05) | 1.82 (-0.09) | 11.13 (+0.66) | 42.4 (-2.2) | 29.3 (+5.7) | … | … | 45.66 (+0.41) | 63.8 (-1.4) | 30.07 (+0.11) | 34.41 (-0.11) | 34.60 (-0.02) | 38.33 (+0.16) |
| ckpt_epoch2 = ckpt_lora_epoch2 (colleague LoRA) | 1.54 (-0.02) | 1.65 (-0.02) | 5.19 (+0.08) | 43.6 (-1.4) | 42.5 (+6.4) | … | … | 45.66 (+2.09) | 54.6 (+0.4) | 24.42 (-0.10) | 33.58 (-0.13) | 30.84 (-0.20) | 38.33 (-0.32) |
| ckpt_fft_epoch2 (colleague full FT) | 1.55 (-0.02) | 1.66 (-0.03) | 5.67 (+0.21) | 44.2 (-0.6) | 28.5 (+6.0) | … | … | 42.22 (+1.15) | 56.8 (-0.8) | 23.22 (+0.12) | 32.64 (+0.02) | 28.99 (-0.08) | 36.26 (-0.15) |

**Table 3 — kit vs original protocol, same model, paired per question** (exact McNemar; letter correctness)

| model | benchmark | n | original only correct | kit only correct | p |
|---|---|---|---|---|---|
| base (untuned Qwen2.5-Omni-7B) | egoschema | 500 | 37 | 30 | 0.464 |
| base (untuned Qwen2.5-Omni-7B) | egocross | 957 | 41 | 45 | 0.747 |
| ckpt_epoch2 = ckpt_lora_epoch2 (colleague LoRA) | egoschema | 500 | 33 | 35 | 0.904 |
| ckpt_epoch2 = ckpt_lora_epoch2 (colleague LoRA) | egocross | 957 | 31 | 51 | 0.0352 |
| ckpt_fft_epoch2 (colleague full FT) | egoschema | 500 | 33 | 29 | 0.704 |
| ckpt_fft_epoch2 (colleague full FT) | egocross | 957 | 42 | 53 | 0.305 |
