# Full fine-tune checkpoint `ckpt_fft_epoch2` (groo_legend/ckpts)

> **Done 2026-09-25 on the H cluster** — evaluated on all five benchmarks (results in the top-level README). How it was run:
> - Downloaded with `ms_dl.py groo_legend/ckpts ckpt_fft_epoch2 <dir> 24` from the H dev box through the institutional proxy
>   (10–26 MB/s per file, ~30 min; sha256 verified against ModelScope).
> - **Loader fix:** the checkpoint is thinker-only (`qwen2_5_omni_thinker`, tensor names without `thinker.`).
>   [`../h_cluster/convert_thinker_ckpt.py`](../h_cluster/convert_thinker_ckpt.py) rewrites each safetensors header with the `thinker.`
>   prefix and copies the tensor bytes verbatim (data sha256 checked), verifies all 1,346 tensors against the base thinker
>   (names / dtypes / shapes), and uses the base's full-Omni thinker-view config (`qwen2_5_omni`, `enable_audio_output: false`) plus the
>   base tokenizer / processor files — the checkpoint's own differ only in serialization and training fields (`padding_side: right`,
>   `use_cache: false`, generation-config token ids); vocab, merges and chat template are identical. LLaMA-Factory then loads it like
>   the base (8,931,813,888 params, no missing or unused weights). Report: [`../h_cluster/fft_epoch2_CONVERSION_REPORT.txt`](../h_cluster/fft_epoch2_CONVERSION_REPORT.txt).
> - **What was trained:** every module differs from the base — LLM layers (333 / 336 tensors), `lm_head`, `embed_tokens`,
>   `audio_tower` (477 / 489), `visual` (460 / 518); the unchanged tensors are norms / biases whose updates round away in bf16.
> - Evaluated as a full model through each benchmark's `make_config.py` with `ADAPTER = model=<dir>`; egoOmni instance `../eval/fft_epoch2/`.
>
> The files below are the 2026-09-23 P-cluster harness, kept for reference.

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
