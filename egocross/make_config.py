#!/usr/bin/env python3
"""LLaMAFactory predict config for EgoCross. Model/template as in the EgoAVU runs; media per the EgoCross protocol
(frame list at the protocol fps, max_pixels 360*480, no audio); greedy. usage: make_config.py TAG ADAPTER|none|model=DIR DATASET FPS OUT"""
import sys
tag, adapter, dataset, fps, out = sys.argv[1:6]
MODEL = "/ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train"
if adapter.startswith("model="):   # a full-weight model (e.g. a full fine-tune) instead of base + LoRA adapter
    MODEL, adapter = adapter[len("model="):], "none"
R = "/ai4good1-shared/liyu/egocross_eval_h"
lines = [
    f"model_name_or_path: {MODEL}",
] + ([f"adapter_name_or_path: {adapter}"] if adapter != "none" else []) + [
    "trust_remote_code: true", "flash_attn: fa2",
    f"video_fps: {fps}", "video_maxlen: 256", "video_max_pixels: 172800", "use_audio_in_video: false",
    "stage: sft", "do_predict: true", "predict_with_generate: true", f"finetuning_type: {'lora' if adapter != 'none' else 'full'}",
    f"eval_dataset: {dataset}", f"dataset_dir: {R}/data", "template: qwen2_omni", "cutoff_len: 32768",
    "overwrite_cache: true", "preprocessing_num_workers: 8", "dataloader_num_workers: 2",
    f"output_dir: {R}/runs/{tag}/{dataset}", "overwrite_output_dir: true", "report_to: none",
    "per_device_eval_batch_size: 1", "bf16: true", "do_sample: false", "max_new_tokens: 512",
    "repetition_penalty: 1.0", "ddp_timeout: 180000000",
]
open(out, "w").write("\n".join(lines) + "\n")
