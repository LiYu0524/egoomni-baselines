#!/usr/bin/env python3
"""Write a LLaMAFactory predict config. Media/template/cutoff settings are the training recipe's
(train_configs/egoavu_lora_r100k_mt_5ep.yaml); decoding is greedy with max_new_tokens 1024 (longest
gold answer = 483 tokens). usage: make_config.py TAG ADAPTER_DIR|none DATASET OUT_YAML"""
import sys
tag, adapter, dataset, out = sys.argv[1:5]
BI = "/ai4good1-shared/liyu/egoavu/bench_infer"
lines = [
    "### model",
    "model_name_or_path: /ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B-thinker-train",
] + ([f"adapter_name_or_path: {adapter}"] if adapter != "none" else []) + [
    "trust_remote_code: true",
    "flash_attn: fa2",
    "video_fps: 2.0",
    "video_maxlen: 64",
    "video_max_pixels: 200704",
    "use_audio_in_video: true",
    "### method",
    "stage: sft",
    "do_predict: true",
    "predict_with_generate: true",
    f"finetuning_type: {'lora' if adapter != 'none' else 'full'}",
    "### dataset",
    f"eval_dataset: {dataset}",
    f"dataset_dir: {BI}/data",
    "template: qwen2_omni",
    "cutoff_len: 32768",
    "overwrite_cache: true",
    "preprocessing_num_workers: 32",
    "dataloader_num_workers: 4",
    "### output",
    f"output_dir: {BI}/runs/{tag}/{dataset}",
    "overwrite_output_dir: true",
    "report_to: none",
    "### eval / generation",
    "per_device_eval_batch_size: 1",
    "bf16: true",
    "do_sample: false",
    "max_new_tokens: 1024",
    "repetition_penalty: 1.0",
    "ddp_timeout: 180000000",
]
open(out, "w").write("\n".join(lines) + "\n")
print(out)
