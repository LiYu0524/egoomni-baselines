# egoOmni baselines

Inference outputs and the evaluation harness for five audio-visual LLM baselines (four open-weight + one API) on the **egoOmni** test set
(`grooLegend/egoOmni` on Hugging Face: 3765 items = 3040 single-turn EN + 725 multi-turn ZH, 5040 answer turns, 1905 clips).

| model | weights | predictions | judge |
|---|---|---|---|
| VideoLLaMA2.1-7B-AV | `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV` (repo branch `audio_visual`) | `eval/preds/videollama2_7b_av/` | ✅ `eval/judgments/`, `eval/results/videollama2_7b_av/` |
| video-SALMONN 2+ 7B | `tsinghua-ee/video-SALMONN2_plus_7B_full` | `eval/preds/salmonn2plus_7b/` | pending |
| video-SALMONN 2+ 72B | `tsinghua-ee/video-SALMONN2_plus_72B_full` | `eval/preds/salmonn2plus_72b/` | pending |
| MiniCPM-o 2.6 (8B) | `openbmb/MiniCPM-o-2_6` | `eval/preds/minicpmo_2_6_8b/` | pending |
| Gemini 3.8 Flash | API (`gemini-3.8-flash`, default media resolution) | `eval/preds/gemini_3_8_flash/` | pending |
| **EgoAVU r100k LoRA (ours)** — Qwen2.5-Omni-7B + LoRA r8, EgoAVU r100k subset, 5 ep | adapter not in this repo | `eval/preds/egoavu_r100k/` | pending |

**Our model — EgoAVU r100k LoRA (added 2026-09-22):** Qwen2.5-Omni-7B thinker + the final (epoch-5) LoRA of the EgoAVU r100k run,
evaluated on the original 3765 items **and** the 136 restored_v3 items in one set: 7176 rows (5176 `gold` + 2000 `self`), 0 errors, 0 empty.
Inference is LLaMAFactory predict with the LoRA's training media settings (2 fps, ≤64 frames, ≤200,704 px/frame, `use_audio_in_video`,
`qwen2_omni` template; audio as 16 kHz mono FLAC as in training) and the same prompts/decoding as the baselines (greedy, ≤256 tokens);
every prediction is joined back with a label/prompt check. Code: `eval/r100k/` (`build_data.py` → `job.sh` per 8-GPU cluster shard → `collect.py`).

**Also here — EgoAVU-Bench:** [`egoavu_bench/`](egoavu_bench/README.md) holds inference outputs (no judging) of two
Qwen2.5-Omni-7B LoRAs (`ckpt_sft`, `ckpt_epoch2`) on all 3,976 EgoAVU-Bench QAs, laid out for the official EgoAVU judge scripts.

All five prediction sets are complete: 7040 rows each (5040 `gold` + 2000 `self` protocol rows). The open-weight runs had 0 inference
errors; the Gemini run has 3 superseded error rows left in place as an audit trail (each key also has a successful row, which is the one
`score.py` uses). Gemini cost **$35.14** for 6315 API requests ($0.0056/request, 34.4M input + 2.5M output/thinking tokens).

## Supplement: `restored_v3` items (added 2026-09-22)

The v9 test file (`test.qa.jsonl`, 3933 rows) = the original 3765 items (content unchanged: questions, answers, options and
multi-turn turns are identical; only evidence-interval metadata and formatting differ, so the predictions above stay valid)
**+ 168 restored items** (138 WI, 30 GP; previously `reject` or `needs_review`, `quality_verified: false`). The restored rows
carry no clip, only a window `original_qa.video_clip_range` in the full Ego4D video, so clips were cut by
`eval/prepare_restored.py`:

| source of the clip | items |
|---|---|
| inside one existing egoOmni clip of the same video | 115 |
| full Ego4D video (13 videos available; 5 cuts use the left half of a 2880-wide side-by-side source, as the dataset clips of those videos do; every cut pixel-verified against an overlapping dataset clip) | 20 |
| stitched from two overlapping clips | 1 |
| **skipped** — window outside every clip and no source video (20 videos) | 32 |

**136 evaluable items**, all five models complete: 136/136 rows each, 0 errors, 90 audio-visual / 46 video-only.
Gemini 3.8 Flash cost for this set: $0.52. Paths: `eval/preds/<tag>/restored_v3/` (72B in 4 cluster slices `p0..p3`),
item file `eval/data/restored_v3/qa_restored_v3.json` (harness schema, `source_kind: restored_single`, id = `sample_id`),
per-item provenance and skip reasons `eval/data/restored_v3/restored_v3_manifest.json`, clip metadata `eval/clips_meta_restored_v3.json`.

Caveats: the file presents each restored item as a standalone single-turn question, and it is evaluated that way, but 56 of the
136 were originally later rounds of a dialogue (`depends_on_earlier_rounds: true`, e.g. the GP items ask "To complete that goal, …");
`minimum_modalities` for these items is derived from `original_qa.loop_annotation.required_modalities` (V 84 / A+V 50 / A 2).
Run against the harness with `EGO_QA_PATH=…/qa_restored_v3.json EGO_CLIPS_META=eval/clips_meta_restored_v3.json`.

## Protocol (details in [`eval/README.md`](eval/README.md))
- Full clip as input; audio fed iff the clip has an audio stream (41 % of clips have none). Each model at its official settings
  (VideoLLaMA2.1: 16 frames + BEATs; SALMONN 2+: 768 frames / 61250 px / 0.1 s, the paper's eval setting; MiniCPM-o 2.6: official omni
  mode, 1-second units of frame + audio, clips > 128 s uniformly subsampled to 128 units; Gemini 3.8 Flash: clip re-encoded to 2 fps /
  ≤1280 px / AAC 64 k and sent inline, default media resolution ≈ 88 tokens per second of clip). Greedy, ≤256 new tokens.
- Prompts: open questions get a one-line "answer briefly" suffix (EN/ZH); MCQ (79 items) = options + "answer with the letter".
- Multi-turn: `gold` = round k conditioned on dataset answers of rounds <k (headline); `self` = conditioned on the model's own answers.
- Scoring: MCQ by letter match; open answers by an LLM judge (`eval/egoomni_eval/judge.py`; local Qwen3-32B via vLLM by default,
  any OpenAI-compatible endpoint via `--backend openai`). `eval/egoomni_eval/score.py` produces the tables.

## Prediction row schema (`eval/preds/<tag>/shard*.jsonl`)
`key` (`<item_id>#r<turn>`), `item_id`, `turn_idx`, `n_turns`, `protocol`, `clip_rel`, `has_audio`, `duration`, `lang`, `fmt`,
`question`, `messages` (the exact chat history sent), `gold`, `options`, `correct_options`, `meta` (category, subcategory, track,
min_modalities, …), `pred`, `error`, `modality_used`, `n_input_tokens`, `latency_s`, `prep_wait_s`, `ts`.
API rows add `usage` (prompt/video/audio/text/output/thoughts tokens), `cost_usd`, `finish_reason`, `api_latency_s`, `media_bytes`,
`media_preset`, `model_version`, `audio_itemized`.

## First result — VideoLLaMA2.1-7B-AV (judge: Qwen3-32B)
overall **20.65 %** (gold) / 18.95 % (self) · single-turn 21.1 % · multi-turn round 20.0 % (gold) / 15.7 % (self) ·
all-rounds-correct 2.1 % · MCQ 68.4 % · open 19.9 %. Breakdowns in `eval/results/videollama2_7b_av/gold.md`.

## Reproducing
`SETUP.md` documents the machine setup (model downloads via `scripts/download_models.sh`, conda envs via `scripts/build_env_*.sh`,
checkpoint validation via `scripts/salmonn_validate_*.py`). Then:
```bash
cd eval && python prepare_clips.py                      # ffprobe → clips_meta.json (already included)
./launch_infer.sh videollama2|salmonn7b|salmonn72b|minicpmo   # 8 GPUs; 72B uses DeepSpeed ZeRO-3 data-parallel
CAP=100 ./launch_api.sh gemini-3.8-flash 24                   # API baseline: 24 HTTP workers + cost_monitor.py (writes STOP_API at the cap)
./launch_judge.sh <tag> && python -m egoomni_eval.score --tag <tag> --protocol gold|self
```
Known deviations from the upstream inference code (GPU-side frame preprocessing for SALMONN, multi-turn prompt cut, ZeRO-3 audio/no-audio
lockstep phases) are documented and quantified in `eval/README.md`.
