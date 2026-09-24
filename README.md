# egoOmni baselines (+ EgoAVU-Bench, EgoCross, EgoSchema)

Inference outputs and evaluation code for four benchmarks:

| benchmark | size | models | where | status |
|---|---|---|---|---|
| **egoOmni** — final test set (`test.qa.jsonl`, 3,882 QAs = 3,765 original + 117 restored) | 3,882 QAs | VideoLLaMA2.1-AV-7B, video-SALMONN 2+ 7B / 72B, MiniCPM-o 2.6, Gemini 3.8 Flash, **EgoAVU LoRAs (ours): r100k, r20k-8gpu, r20k-32gpu** | [`eval/`](eval/README.md) | predictions for **3,877 / 3,882** QAs on all models (5 restored QAs have no video); judged: VideoLLaMA2.1-AV-7B and **the three EgoAVU LoRAs** |
| **EgoAVU-Bench** | 3,976 QAs, 500 videos | two Qwen2.5-Omni-7B LoRAs (`ckpt_sft`, `ckpt_epoch2`) + untuned Qwen2.5-Omni-7B control | [`egoavu_bench/`](egoavu_bench/README.md) | predictions complete (3,976 / 3,976 each); **v2: scored to the official standard + untuned-base control** ([`results/`](egoavu_bench/results/README.md)) — neither LoRA beats the base except `ckpt_epoch2` on hallucination probes; **v3 (2026-09-24): + the three EgoAVU LoRAs (ours)**, which beat the base on every judged category |
| **EgoCross** (closed set) | 957 MCQs, 4 domains | untuned Qwen2.5-Omni-7B + the three EgoAVU LoRAs | [`egocross/`](egocross/README.md) | done; predictions (Codabench format) + aggregate scores only — the test answers are hidden |
| **EgoSchema** (public Subset) | 500 five-way MCQs | untuned Qwen2.5-Omni-7B + the three EgoAVU LoRAs + the two colleague LoRAs (`ckpt_sft`, `ckpt_epoch2`) | [`egoschema/`](egoschema/README.md) | done (full 5,031-question set not run: answers are server-side) |

Every problem hit and how it was handled: [`docs/ISSUES_AND_HANDLING.md`](docs/ISSUES_AND_HANDLING.md).

## EgoAVU LoRAs (ours) — results on four benchmarks (2026-09-24)

Three LoRA fine-tunes of **Qwen2.5-Omni-7B (Thinker)** on EgoAVU window groups (LoRA r8/α16, 5 epochs; adapters are in private
ModelScope repos, not here): **r100k** (100k-row subset, 8 GPUs), **r20k-8gpu** and **r20k-32gpu** (20k-row subset on 8 / 32 GPUs).
"base" is the untuned Qwen2.5-Omni-7B through the identical pipeline. All runs: H cluster, H200, LLaMA-Factory predict with each
benchmark's protocol, greedy decoding; scripts in [`h_cluster/`](h_cluster/README.md).

**EgoAVU-Bench** (official protocol; judge Qwen3-235B-A22B-Instruct-2507; S = judge 1–5, M/R = METEOR/ROUGE-L; details
[`egoavu_bench/results/tables.md`](egoavu_bench/results/tables.md))

| model | SSA S | AVDN S | AVDN M | AVDN R | AVSN S | AVSN M | AVSN R | TR Acc | AVH Acc |
|---|---|---|---|---|---|---|---|---|---|
| base | 1.50 | 1.78 | 15.93 | 15.53 | 1.92 | 10.47 | 15.30 | 44.6 | 23.6 |
| ckpt_sft | 1.53 | 1.73 | 7.99 | 11.92 | 1.68 | 4.39 | 9.75 | 43.6 | 13.0 |
| ckpt_epoch2 | 1.56 | 1.70 | 4.93 | 9.91 | 1.67 | 5.11 | 10.76 | 45.0 | 36.1 |
| **r20k-32gpu** | 2.62 | 2.03 | 12.57 | 16.76 | 2.06 | 19.57 | 24.76 | 47.2 | 97.1 |
| **r100k** | 2.65 | 2.01 | 11.97 | 16.82 | 2.01 | 20.59 | 26.26 | 44.2 | 98.4 |
| **r20k-8gpu** | 2.65 | 2.02 | 12.71 | 16.92 | 2.05 | 19.33 | 24.83 | 42.0 | 97.5 |
| paper: base / paper LoRA | 1.50 / 3.15 | 2.37 / 2.60 | 10.69 / 12.20 | 14.74 / 17.19 | 1.99 / 2.45 | 9.99 / 22.53 | 13.39 / 28.34 | 53.2 / 64.3 | 42.7 / 61.7 |

All three LoRAs beat the base on every judged category (paired 95% CIs exclude 0); TR accuracy differences are not significant.
AVH yes/no: 290 of the 304 probes have the gold answer "No" (always-"No" = 95.4); the LoRAs' gain is the removal of the base's
"Yes" bias (base says "Yes" to 80.9% of probes, LoRAs to 3–6%).

**egoOmni** (this repo's benchmark; judge Qwen3-32B; 5,176 gold rows = 3,765 original + 136 restored items; `eval/results/`)

| model | gold overall | gold single-turn | gold multi-turn round | gold all-rounds-correct | self overall | self multi-turn round |
|---|---|---|---|---|---|---|
| **r20k-8gpu** | **30.12** | 28.94 | **32.00** | **5.93** | **28.77** | **28.50** |
| r20k-32gpu | 29.56 | **29.35** | 29.90 | 5.10 | 27.94 | 25.70 |
| r100k | 28.69 | 28.05 | 29.70 | 5.52 | 26.60 | 24.30 |
| VideoLLaMA2.1-AV-7B (original items only) | 20.65 | 21.12 | 19.95 | 2.07 | 18.95 | 15.65 |

Final-bench-only scores (`eval/results_final/`, 5,152 rows) differ by ≤ 0.1 pt. r20k-32gpu was first run on the P cluster
(29.52 / 27.57); the H rerun gives 29.56 / 27.94. The r100k predictions from the P run were replaced by the judged H rerun (8 shards).

**EgoCross** (closed set, official Codabench scoring; [`egocross/`](egocross/README.md))

| model | overall | Surgery | Industry | XSports | Animal | vs base (McNemar) |
|---|---|---|---|---|---|---|
| base | 45.25 | 43.1 | 46.5 | 44.3 | 48.1 | — |
| r100k | 43.89 | 40.3 | 45.3 | 45.5 | 45.4 | −1.36, p = 0.40 |
| r20k-8gpu | 45.66 | 41.7 | 43.7 | 50.0 | 48.6 | +0.42, p = 0.82 |
| r20k-32gpu | 45.14 | 39.2 | 44.9 | 50.4 | 47.5 | −0.10, p = 1.00 |

**EgoSchema** (public 500-question Subset, lmms-eval prompt; the videos have no audio; [`egoschema/`](egoschema/README.md))

| model | accuracy | vs base (McNemar) |
|---|---|---|
| base | **65.2** | — |
| r20k-8gpu | 64.0 | −1.2, p = 0.59 |
| r20k-32gpu | 60.4 | −4.8, p = 0.02 |
| r100k | 58.0 | −7.2, p = 0.0006 |
| ckpt_sft (colleague) | 56.4 | −8.8, p < 0.0001 |
| ckpt_epoch2 (colleague) | 54.2 | −11.0, p < 0.0001 |

Summary: large in-domain gains (EgoAVU-Bench), ~29–30% on egoOmni, no change on EgoCross, and a loss on long-form, video-only
EgoSchema that grows with the amount of EgoAVU training; the two colleague LoRAs lose even more (−8.8 / −11.0).

## egoOmni

The original egoOmni test set (`grooLegend/egoOmni` on Hugging Face) has 3765 items = 3040 single-turn EN + 725 multi-turn ZH,
5040 answer turns, 1905 clips; the final test file adds 117 restored items (see [below](#supplement-restored_v3-items-added-2026-09-22)).

| model | weights | predictions | judge |
|---|---|---|---|
| VideoLLaMA2.1-7B-AV | `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV` (repo branch `audio_visual`) | `eval/preds/videollama2_7b_av/` | ✅ `eval/judgments/`, `eval/results/videollama2_7b_av/` |
| video-SALMONN 2+ 7B | `tsinghua-ee/video-SALMONN2_plus_7B_full` | `eval/preds/salmonn2plus_7b/` | pending |
| video-SALMONN 2+ 72B | `tsinghua-ee/video-SALMONN2_plus_72B_full` | `eval/preds/salmonn2plus_72b/` | pending |
| MiniCPM-o 2.6 (8B) | `openbmb/MiniCPM-o-2_6` | `eval/preds/minicpmo_2_6_8b/` | pending |
| Gemini 3.8 Flash | API (`gemini-3.8-flash`, default media resolution) | `eval/preds/gemini_3_8_flash/` | pending |
| **EgoAVU r100k LoRA (ours)** — Qwen2.5-Omni-7B + LoRA r8, EgoAVU r100k subset, 5 ep | adapter not in this repo | `eval/preds/egoavu_r100k/` | ✅ `eval/judgments/`, `eval/results/egoavu_r100k/` |
| **EgoAVU r20k-8gpu LoRA (ours)** — same recipe, r20k subset, 8 GPUs | adapter not in this repo | `eval/preds/egoavu_r20k8g/` | ✅ `eval/results/egoavu_r20k8g/` |
| **EgoAVU r20k-32gpu LoRA (ours)** — same recipe, r20k subset, 32 GPUs (global batch 32, lr 2.83e-5) | adapter not in this repo | `eval/preds/egoavu_r20k32g/` | ✅ `eval/results/egoavu_r20k32g/` |

> **Problems and how they were handled:** every issue hit during setup and inference — blocked downloads, dependency conflicts,
> silent audio drop, SALMONN checkpoint choice, a ZeRO-3 deadlock, the Gemini usage-report quirk, the unjudged models, and the
> restored items — is written up in [`docs/ISSUES_AND_HANDLING.md`](docs/ISSUES_AND_HANDLING.md).
>
> **5 QAs of the final test set have no results for any model** (restored items whose video is not available); see
> [the section below](#restored-items-without-results) and the full list in the doc.

**Our model — EgoAVU r100k LoRA (added 2026-09-22):** Qwen2.5-Omni-7B thinker + the final (epoch-5) LoRA of the EgoAVU r100k run,
evaluated on the original 3765 items **and** the 136 restored_v3 items in one set: 7176 rows (5176 `gold` + 2000 `self`), 0 errors, 0 empty.
Inference is LLaMAFactory predict with the LoRA's training media settings (2 fps, ≤64 frames, ≤200,704 px/frame, `use_audio_in_video`,
`qwen2_omni` template; audio as 16 kHz mono FLAC as in training) and the same prompts/decoding as the baselines (greedy, ≤256 tokens);
every prediction is joined back with a label/prompt check. Code: `eval/r100k/` (`build_data.py` → `job.sh` per 8-GPU cluster shard → `collect.py`).

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
| **skipped — no results for any model** (window outside every clip, no source video; 20 videos) | 32 |

**136 evaluable items**, all six models complete: 136/136 rows each, 0 errors, 90 audio-visual / 46 video-only (EgoAVU r100k
keeps them inside its main shard files).
Gemini 3.8 Flash cost for this set: $0.52. Paths: `eval/preds/<tag>/restored_v3/` (72B in 4 cluster slices `p0..p3`),
item file `eval/data/restored_v3/qa_restored_v3.json` (harness schema, `source_kind: restored_single`, id = `sample_id`),
per-item provenance and skip reasons `eval/data/restored_v3/restored_v3_manifest.json`, clip metadata `eval/clips_meta_restored_v3.json`.

Caveats: the file presents each restored item as a standalone single-turn question, and it is evaluated that way, but 56 of the
136 were originally later rounds of a dialogue (`depends_on_earlier_rounds: true`, e.g. the GP items ask "To complete that goal, …");
`minimum_modalities` for these items is derived from `original_qa.loop_annotation.required_modalities` (V 84 / A+V 50 / A 2).
Run against the harness with `EGO_QA_PATH=…/qa_restored_v3.json EGO_CLIPS_META=eval/clips_meta_restored_v3.json`.

### Final test set vs. what was run
The final `test.qa.jsonl` (3,882 rows, received after this run) keeps the 3,765 original items unchanged and **117 of the 168**
restored items. Against that final set:

| restored items | count | predictions |
|---|---|---|
| in the final set, evaluated | 112 | all six models |
| **in the final set, no video available** | **5** | **none** |
| not in the final set, but evaluated earlier (20 cut from full Ego4D videos, 3 from single clips, 1 stitched) | 24 | kept in the repo, flagged `in_final_bench: false` — exclude them when scoring the final set |
| not in the final set and never evaluable | 27 | none |

`eval/data/restored_v3/final_bench_restored_ids.json` lists the 117 final restored ids, and every row of
`eval/data/restored_v3/restored_v3_manifest.json` carries `in_final_bench`.

### Restored items without results
The 32 restored items we could not evaluate ask about a time window of the original Ego4D video that lies partly (21 items,
0.9–124 s missing) or entirely (11 items) outside every clip in the egoOmni dataset. The full Ego4D videos for those 20 videos
are neither on Hugging Face nor on our cluster, and the Ego4D download credentials we tried are no longer valid, so the clips
could not be cut. Cutting only the covered part was rejected because it could remove the evidence the question asks about.
**27 of the 32 were dropped from the final test set; 5 are still in it and have no results for any model:**
- `5f77b76b-24d9-4489-9561-861fc66a1917__0030` — window 550.46–582.46 s, 32.00 s not covered by any clip
- `c398a6bf-58cf-4318-9b98-babf649827e5__0028` — window 41.38–73.38 s, 0.85 s not covered by any clip
- `eab3bfc0-d611-4a40-89b4-7f9747b55e6a__0026` — window 895.91–927.91 s, 32.00 s not covered by any clip
- `eab3bfc0-d611-4a40-89b4-7f9747b55e6a__0027` — window 693.52–725.52 s, 1.95 s not covered by any clip
- `eab3bfc0-d611-4a40-89b4-7f9747b55e6a__0028` — window 693.52–725.52 s, 1.95 s not covered by any clip

All 32 are listed in [`docs/ISSUES_AND_HANDLING.md` §8](docs/ISSUES_AND_HANDLING.md#the-32-items-without-results). To finish the
5, put the full videos under `ego4d_full/<video_id>.mp4` and rerun `eval/prepare_restored.py` and the inference scripts (resumable).

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

## EgoAVU-Bench (`egoavu_bench/`)

Raw predictions of two LoRA fine-tunes of **Qwen2.5-Omni-7B (Thinker)** on all **3,976 QAs** of EgoAVU-Bench (500 videos,
7 categories) — **inference only, nothing judged**; laid out for the official EgoAVU evaluation scripts. Full details:
[`egoavu_bench/README.md`](egoavu_bench/README.md).

| file | model | rows | empty | hit the 1,024-token cap |
|---|---|---|---|---|
| `egoavu_bench/predictions/ckpt_sft.json` | LoRA `ckpt_sft` | 3,976 | 0 | 18 |
| `egoavu_bench/predictions/ckpt_epoch2.json` | LoRA `ckpt_epoch2` | 3,976 | 0 | 5 |

- **Protocol:** LLaMA-Factory predict; each QA sees only its own `[start_time, end_time]` window; video 2 fps, ≤64 frames,
  ≤200,704 px; 16 kHz audio interleaved with video (`use_audio_in_video`); greedy, ≤1,024 new tokens; every prediction joined
  back to its bench row and verified (3,976 / 3,976 for both models).
- **Scoring with the official scripts needs two fixes:** `llm_as_judge.py` crashes on every input as published (`str.format`
  on a prompt containing a literal JSON block → escape those two braces as `{{ }}`); `captioning_eval.py`'s default categories
  select nothing — pass the full category names, one narration category per run. Join on `bench_idx`, never on `question`
  alone (346 question strings repeat).
- **Input caveats (from the source data, not the pipeline):** 106 windows are fully black (90 also silent), 35 partly black;
  24 windows are shorter than 5 s; Qwen2.5-Omni keeps only the first 300 s of audio, so 822 longer windows are video-only at
  the tail. Lists in `egoavu_bench/notes/bench_window_notes.json`.
