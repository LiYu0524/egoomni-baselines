# EgoAVU-Bench — 2026-09-25 runs

Scored exactly like v3 (official judge `judge_official.py` with Qwen3-235B-A22B-Instruct-2507 on 4 H200, official
`captioning_eval.py`, `closed_ended_acc.py`, `summarize.py`), each run judged in its own directory so earlier runs were not re-judged.

| directory | model | protocol | predictions |
|---|---|---|---|
| `original_protocol_ckpt_fft_epoch2/` | `ckpt_fft_epoch2` (groo_legend/ckpts full fine-tune) | original: LLaMA-Factory, 2 fps ≤ 64 frames, ≤ 200,704 px, audio-in-video, 1,024 new tokens | `../../predictions/ckpt_fft_epoch2.json` |
| `kit_protocol_base/` | untuned Qwen2.5-Omni-7B | colleague's EgoToM kit: native transformers + `qwen_omni_utils`, 2 fps ≤ 360 frames, ≤ 156,800 px, audio-in-video, 1,024 new tokens | `../../predictions/kit_base.json` |
| `kit_protocol_ckpt_epoch2/` | `ckpt_epoch2` = groo_legend/ckpts `qwen2.5omni7b/ckpt_lora_epoch2` (merged into the weights) | same | `../../predictions/kit_ckpt_epoch2.json` |
| `kit_protocol_ckpt_fft_epoch2/` | `ckpt_fft_epoch2` | same | `../../predictions/kit_ckpt_fft_epoch2.json` |

Each directory: `summary.json` / `tables.md` (all models judged so far, incl. paired differences), `judge_scores.csv`,
`judge_counts.json`, `judge_items_<tag>.jsonl` (per-item judge output), `caption_*.csv`, `closed_ended_acc_<tag>.json`.

Kit-protocol inference: `h_cluster/nat_predict_lf.py` (resumable, one process per shard) over 30 interleaved shards
(`bench_idx % 30`); the base run's unfinished tail was rebalanced over 64 shards (`h_cluster/split_kit_rest.py`,
`finalize_kit_rest.py`) to finish before the cards were handed back. The 2 windows shorter than 1 s were widened to 1 s for the
video (the reader needs ≥ 2 frames); their audio keeps the original window.
