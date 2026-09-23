#!/usr/bin/env python
"""LLaMAFactory predict config for a FULL fine-tuned Qwen2.5-Omni thinker on EgoAVU-Bench.
Media/template settings are the EgoAVU training recipe (2 fps, ≤64 frames, ≤200,704 px, use_audio_in_video,
qwen2_omni, cutoff 32768); decoding greedy, max_new_tokens 1024 — identical to the ckpt_sft / ckpt_epoch2 runs in
bench_infer/tools/make_config.py, so the three models are directly comparable.
usage: make_config_fft.py TAG MODEL_DIR DATASET OUT_YAML"""
import sys
tag, model, dataset, out = sys.argv[1:5]
BI = "/ai4good1-shared/liyu/egoavu/bench_infer"          # datasets are reused read-only from the existing harness
R = "/ai4good1-shared/liyu/egoavu/bench_infer_fft"
open(out, "w").write("\n".join([
    "### model", f"model_name_or_path: {model}", "trust_remote_code: true", "flash_attn: fa2",
    "video_fps: 2.0", "video_maxlen: 64", "video_max_pixels: 200704", "use_audio_in_video: true",
    "### method", "stage: sft", "do_predict: true", "predict_with_generate: true", "finetuning_type: full",
    "### dataset", f"eval_dataset: {dataset}", f"dataset_dir: {BI}/data", "template: qwen2_omni", "cutoff_len: 32768",
    "overwrite_cache: true", "preprocessing_num_workers: 32", "dataloader_num_workers: 8",
    "### output", f"output_dir: {R}/runs/{tag}/{dataset}", "overwrite_output_dir: true", "report_to: none",
    "### eval / generation", "per_device_eval_batch_size: 1", "bf16: true", "do_sample: false",
    "max_new_tokens: 1024", "repetition_penalty: 1.0", "ddp_timeout: 180000000"]) + "\n")
print(out)
