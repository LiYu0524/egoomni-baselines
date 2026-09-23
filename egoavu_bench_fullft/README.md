# EgoAVU-Bench — full fine-tune checkpoint (prepared, not yet run)

Harness for evaluating a **full fine-tune** of Qwen2.5-Omni on EgoAVU-Bench, next to the LoRA runs in
[`../egoavu_bench/`](../egoavu_bench/README.md). Prepared 2026-09-23; the run itself did not happen (the GPUs were
returned before the 17.9 GB checkpoint finished downloading), so there are no predictions here yet.

**Target checkpoint:** `groo_legend/ckpts`, subdir `ckpt_fft_epoch2` — 4 shards, 17.88 GB, config
`model_type: qwen2_5_omni_thinker` / `Qwen2_5OmniThinkerForConditionalGeneration`, bf16.

## Two things to know before running it
1. **Only `www.modelscope.ai` serves that repo** (`www.modelscope.cn` answers "record not found"), and the P-cluster boxes
   reach it only through the institutional HTTP proxy with `.modelscope.*` removed from `no_proxy`. That proxy caps
   **per account, not per connection**: 1 stream ≈ 0.3 MB/s, 8 ≈ 4 MB/s, 24 ≈ 6 MB/s, and downloaders on three boxes do not
   add up — budget ~50 min for 17.9 GB. `ms_dl.py` does the parallel, resumable, sha256-verified download:
   `ms_dl.py groo_legend/ckpts ckpt_fft_epoch2 <outdir> 24`.
2. **The checkpoint is thinker-only, and transformers 4.54 has no Auto mapping for it** (`MODEL_FOR_TEXT_TO_WAVEFORM` knows
   only `qwen2_5_omni` → `Qwen2_5OmniForConditionalGeneration`; the LoRA runs used the full-Omni *thinker view*
   `models/Qwen2.5-Omni-7B-thinker-train`). Expect LLaMAFactory to fail to load it as published — wrap the config as a full
   Omni model (weight keys then need a `thinker.` prefix) or load `Qwen2_5OmniThinkerForConditionalGeneration` explicitly.
   Verify on one shard before launching the fleet.

## Files
| file | role |
|---|---|
| `ms_dl.py` | ModelScope downloader described above (no credentials; uses the proxy env vars) |
| `make_config_fft.py` | LLaMAFactory predict config: `finetuning_type: full`, the EgoAVU training media settings (2 fps, ≤64 frames, ≤200,704 px, `use_audio_in_video`, `qwen2_omni`, cutoff 32,768), greedy, `max_new_tokens 1024` — identical decoding to the LoRA runs, so the numbers are comparable |
| `collect_fft.py` | joins `generated_predictions.jsonl` back to the bench rows and asserts label == gold answer and question ∈ prompt (same checks as `egoavu_bench/tools/collect_predictions.py`) |
| `job_fft.sh` | one clusterx job: waits for a `MODEL_READY` marker, restores the EgoAVU env tarball, runs predict on one shard, collects |

## Plan that was ready to go
4 clusterx jobs × 8 GPUs, one bench shard each (`egoavu_bench_eval_s{k}of4`, 994 rows), ~20–25 min total including env restore
and preprocessing. Paths inside the scripts are the P cluster's (`/ai4good1-shared/liyu/egoavu/...`) and need editing on a new cluster.
