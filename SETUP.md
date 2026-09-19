# egoOmni baselines — dev box (8×A100-80GB)

Set up 2026-09-18. Baselines for the egoOmni benchmark (`/ai4good1-shared/liyu/egoOmni`, 73 GB, 1905 clips, `test/qa.json` 3765 items; ~70 % of clips have AAC audio, ~30 % are video-only).

## Layout
```
/ai4good1-shared/liyu/egoOmni_baselines/        persistent PVC
├── env.sh            source in every script: PATH, tuna pip index, HF/ModelScope env, EGO_ENVS
├── .proxy_env        institutional proxy creds (chmod 600, never committed) — source only for huggingface.co / github.com
├── miniconda3/       base (py3.14) + env `dl` (modelscope, hf CLI)
├── envs/             *.tar snapshots of the runtime envs (see scripts/persist_envs.sh)
├── models/
│   ├── VideoLLaMA2.1-7B-AV/            HF DAMO-NLP-SG/VideoLLaMA2.1-7B-AV via ModelScope (config.json patched → local siglip; original in config.json.orig)
│   ├── siglip-so400m-patch14-384/      vision tower/processor for VideoLLaMA2
│   ├── Qwen2.5-VL-7B-Instruct/  Qwen2.5-VL-72B-Instruct/  whisper-large-v3/     SALMONN 2+ bases (ModelScope)
│   ├── video-SALMONN2_plus_{7B,72B}_full/       ★ official merged releases — USE THESE FOR EVAL (HF tsinghua-ee, via proxy)
│   ├── video_SALMONN2plus_{7B,72B}_audioAlign/  audio-aligned bases (for fine-tuning; also the LoRA's base)
│   └── video-SALMONN-2_plus_{7B,72B}/           released LoRA ckpts (train/DPO reproduction)
├── repos/VideoLLaMA2 (branch audio_visual @ 8cedfbd)   repos/video-SALMONN-2 (main @ 2d28e84, use video_SALMONN2_plus/)
├── scripts/  logs/
/shared/egoOmni_envs/{videollama2,salmonn2plus}   runtime conda envs on local NVMe (container-local!)
```

## Why envs are on NVMe
The PVC is ~85× slower than NVMe for small-file ops (2.8 ms vs 0.03 ms per file); a torch env is ~50k files.
Envs therefore live on `/shared` (fast, but wiped on container rebuild) and are snapshotted as single tarballs on the PVC:
```bash
scripts/persist_envs.sh videollama2 salmonn2plus   # after changing an env
scripts/restore_envs.sh                            # after a container rebuild
```

## Network routing (measured)
| route | status |
|---|---|
| huggingface.co direct | blocked |
| hf-mirror.com | metadata only — LFS redirect (cas-bridge.xethub.hf.co) blocked, weights never arrive |
| ModelScope direct | ~45 MB/s, no proxy — first choice |
| institutional HTTP proxy (credentials in `.proxy_env`, not committed) | 2 MB/s 09:00–21:00, 20 MB/s 21:00–09:00; needed for HF repos not on ModelScope (tsinghua-ee/*) and the egoOmni dataset |

## Envs
| env | key pins | notes |
|---|---|---|
| `videollama2` | torch 2.2.0+cu118, transformers 4.42.3, flash-attn 2.5.8 (prebuilt wheel), opencv 4.5.5.64, `pip install -e repos/VideoLLaMA2` | exactly repo requirements.txt + README steps |
| `salmonn2plus` | torch 2.7.1, transformers 4.51.3, peft 0.15.2, liger-kernel 0.5.10, deepspeed 0.16.0, flash-attn **2.8.0.post2** (prebuilt for torch 2.7), torchaudio **2.7.1** | repo pins flash_attn 2.7.4.post1 (no torch-2.7 wheel; API-compatible) and torchaudio 2.5.1 (incompatible with torch 2.7.1) |

## Usage
```bash
source /ai4good1-shared/liyu/egoOmni_baselines/env.sh
# VideoLLaMA2.1-7B-AV  (audio-visual: --modal av | v | a)
CUDA_VISIBLE_DEVICES=0 $EGO_ENVS/videollama2/bin/python scripts/videollama2_infer.py --model models/VideoLLaMA2.1-7B-AV --video clip.mp4 --modal av
# video-SALMONN 2+ 7B / 72B (paper eval settings: --max_frames 768 --max_pixels 61250 --interval 0.1)
CUDA_VISIBLE_DEVICES=0 $EGO_ENVS/salmonn2plus/bin/python scripts/salmonn_infer.py --model models/video-SALMONN2_plus_7B_full --video clip.mp4
# 72B: bf16 weights ≈ 145 GB → shard across GPUs (needs ~150 GB weights + ~10 GB activations free in total):
$EGO_ENVS/salmonn2plus/bin/python scripts/salmonn_infer.py --model models/video-SALMONN2_plus_72B_full --video clip.mp4 --device_map auto
# (or the repo's scripts/test_8.sh: deepspeed zero3 over 8 GPUs)
```
### Why `_full` and not the README's `--lora_ckpt` recipe (verified 2026-09-19, `scripts/salmonn_validate_*.py`)
Rebuilding every tensor of `_full` 7B shard 1 (956 tensors) from `_audioAlign` + LoRA (`base + B@A·α/r`, `modules_to_save` for qformer/audio_proj/merger) matches the official release **bit-for-bit except one tensor: `audio.q_tokens`** (the Q-Former query, shape 1×1×3072, rel. max err 0.42). `q_tokens` is a bare `nn.Parameter`, so peft's `modules_to_save` silently cannot store it and the adapter never carried it; the `_audioAlign` upload's value is not the one the released model uses. A `gen_audio_model.py` base is worse still (random `q_tokens`). Hence eval = `_full`; the repo's `scripts/test.sh` runs on it with `--model <full> --model_base <full>` and no `--lora_ckpt`. `scripts/salmonn_merge_lora.py` remains for reproducing the merge (and shows the discrepancy).

## Status (2026-09-19 02:00 CST)
| model | weights | env | smoke test on egoOmni clip |
|---|---|---|---|
| VideoLLaMA2.1-7B-AV | ✅ | ✅ | ✅ AV inference, 16.4 GiB, audio used |
| video-SALMONN 2+ 7B (`_full`) | ✅ | ✅ | ✅ AV inference, 17.2 GiB, 27 550 tokens, audio described |
| video-SALMONN 2+ 72B (`_full`) | ✅ | ✅ | ⏸ loads/shards fine (`--device_map auto`, 32 shards); generate OOMs while the agentdog job holds ~66 GB/card — rerun when GPUs free up |
Also on disk: `_audioAlign` 7B/72B + LoRA 7B/72B (training reproduction); `Qwen2.5-VL-{7B,72B}-Instruct` + `whisper-large-v3` (only for `gen_audio_model.py`; 166 GB, safe to delete if not training from scratch).

## Gotchas
- VideoLLaMA2 `processor['video'](path, va=True)` will fail on the ~30 % clips without an audio stream — guard with ffprobe or fall back to `va=False`.
- All 8 GPUs currently carry ~37 GB from another job (agentdog); ~44 GB free per card is enough for the 7B models, not for 72B.
- `pkill -f` inside `ssh 'bash -c …'` matches the wrapper's own cmdline — anchor patterns with `^`.
