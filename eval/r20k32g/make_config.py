#!/usr/bin/env python
"""LLaMAFactory predict config = the r100k training recipe's media/template settings (train_configs/egoavu_lora_r100k_mt_5ep.yaml:
2 fps, ≤64 frames, ≤200,704 px, use_audio_in_video, qwen2_omni, cutoff 32768) + greedy decoding with max_new_tokens 256
(the budget every other egoOmni baseline got).  usage: make_config.py DATASET OUT_YAML"""
import sys
ds, out = sys.argv[1:3]
R = "/ai4good1-shared/liyu/egoOmni_baselines/eval/r20k32g"
open(out, "w").write("\n".join([
    "model_name_or_path: /ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train",
    "adapter_name_or_path: /ai4good1-shared/liyu/egoavu/release/EgoAVU-Qwen2.5-Omni-7B-LoRA-r20k/32gpu",
    "trust_remote_code: true", "flash_attn: fa2",
    "video_fps: 2.0", "video_maxlen: 64", "video_max_pixels: 200704", "use_audio_in_video: true",
    "stage: sft", "do_predict: true", "predict_with_generate: true", "finetuning_type: lora",
    f"eval_dataset: {ds}", f"dataset_dir: {R}/data", "template: qwen2_omni", "cutoff_len: 32768",
    "overwrite_cache: true", "preprocessing_num_workers: 32", "dataloader_num_workers: 8",
    f"output_dir: {R}/runs/{ds}", "overwrite_output_dir: true", "report_to: none",
    "per_device_eval_batch_size: 1", "bf16: true", "do_sample: false", "max_new_tokens: 256", "repetition_penalty: 1.0",
    "ddp_timeout: 180000000"]) + "\n")
print(out)
