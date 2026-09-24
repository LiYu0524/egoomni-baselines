# EgoAVU-Bench — two Qwen2.5-Omni-7B LoRAs: predictions + official-standard scores

Raw predictions of two LoRA fine-tunes of **Qwen2.5-Omni-7B (Thinker)** on all **3,976 QAs** of EgoAVU-Bench
(`egoavu_bench_combined.csv`, 500 videos, 7 categories, sha256 `7d4ca546b1838e44068796a2665f25ce567d447f87b37e787e871e5d3757cb3a`).
**v2 (2026-09-23): scored to the official EgoAVU standard, with a base-model control** — details, CIs and per-item files in
[`results/`](results/README.md).

| model | SSA S | AVDN S | AVDN M | AVDN R | AVSN S | AVSN M | AVSN R | TR Acc | AVH Acc* |
|---|---|---|---|---|---|---|---|---|---|
| **base_qwen25omni7b** | 1.50 | 1.78 | 15.93 | 15.53 | 1.92 | 10.47 | 15.30 | 44.60 | 23.62 |
| **ckpt_sft** | 1.53 | 1.73 | 7.99 | 11.92 | 1.68 | 4.39 | 9.75 | 43.60 | 12.98 |
| **ckpt_epoch2** | 1.56 | 1.70 | 4.93 | 9.91 | 1.67 | 5.11 | 10.76 | 45.00 | 36.10 |
| paper: Qwen2.5-Omni-7B (base) | 1.50 | 2.37 | 10.69 | 14.74 | 1.99 | 9.99 | 13.39 | 53.20 | 42.69 |
| paper: Ours (LoRA) | 3.15 | 2.60 | 12.20 | 17.19 | 2.45 | 22.53 | 28.34 | 64.31 | 61.69 |

\* AVH = the 304 yes/no probes (macro over subtypes); the paper's AVH also scores the open-ended AVH items by an unstated rule —
calibrated variant and discussion in [`results/README.md`](results/README.md).

**The control run makes the comparison meaningful.** The untuned Qwen2.5-Omni-7B, through the identical pipeline, reproduces the
paper's base closely on the unambiguous metrics (SSA 1.502 vs 1.50, AVSN S 1.917 vs 1.99, AVSN M 10.47 vs 9.99), so the pipeline is
sound. Against that control, **neither LoRA improves**: AVSN judge S drops significantly for both (−0.24), METEOR/ROUGE-L collapse
because the fine-tuned answers are far shorter, and TR accuracy is unchanged. The one gain is hallucination behaviour in
`ckpt_epoch2` (AVH yes/no 36.1 vs 23.6 for the base), while `ckpt_sft` is worse (13.0; it answers "Yes" to 91 % of probes).

The prediction files below are laid out for the official EgoAVU evaluation scripts.

| file | model | rows | empty | hit 1,024-token cap |
|---|---|---|---|---|
| `predictions/ckpt_sft.json` | LoRA `ckpt_sft` | 3,976 | 0 | 18 |
| `predictions/ckpt_epoch2.json` | LoRA `ckpt_epoch2` | 3,976 | 0 | 5 |

## Scoring with the official EgoAVU code (github.com/facebookresearch/EgoAVU, `evaluation/`, commit 15bd5bb)
`predictions/` holds only these two files, so it can be passed as the input directory as is.
- **`llm_as_judge.py` crashes on every input as published**: `JUDGE_PROMPT` contains the literal JSON block
  `{ "rating": …, "reason": … }` and is filled with `str.format`, which raises `KeyError('\n  "rating"')` before anything is judged.
  Fix: in the *Output Format* block write `{{` and `}}` for those two braces (nothing else changes). Then:
  `python evaluation/llm_as_judge.py --input_dir egoavu_bench/predictions --output_csv judge.csv --temperature 0.0 --max_new_tokens 512`
- **`captioning_eval.py`** filters on `category`, which here holds the bench CSV's full names; its default `--categories avsn avdn`
  therefore selects nothing (silently, no CSV written). Run it once per narration category so the two are not pooled:
  `--categories "Audio-Visual Segment Narration"` (1,200 items) and `--categories "Audio-Visual Dense Narration"` (500 items),
  each with its own `--output_csv` outside `predictions/`. Note that it scores CIDEr one sample at a time; pycocoevalcap's CIDEr
  takes IDF from the corpus it is given, so per-sample CIDEr is 0 by construction — rely on METEOR / ROUGE-L.
- Join key: `bench_idx` (= 0-based data row of the CSV) or `(video_id, question)`. Never join on `question` alone:
  346 question strings occur more than once (1,277 rows) and 4 (question, answer) pairs repeat across different videos.

## Item schema (JSON list per model, bench order, ASCII-only)
| field | meaning |
|---|---|
| `bench_idx` | 0-based CSV data row, 0..3975, each exactly once |
| `video_id`, `start_time`, `end_time`, `category` | from the bench CSV |
| `question`, `answer` | bench question / gold answer (identical to the CSV; whitespace-stripped, none had any) |
| `output` | model prediction, unmodified |
| `category_short` | avsn / avdn / ssa / tr / avh_action / avh_object / avh_sound (informational) |
| `output_tokens`, `truncated` | prediction length in Qwen tokens; `truncated` = hit the 1,024-token cap |
| `n_video_tokens`, `n_audio_tokens`, `audio_seconds_fed` | exact multimodal token counts in the prompt the model received (audio = 25 tokens/s) |
| `audio_used`, `audio_capped_at_300s` | audio fed at all; window longer than the 300-s audio limit (see protocol) |
| `video_black_frames_of_8`, `video_all_black`, `audio_silent` | measured on the exact input window: black = mean luma < 2 of 255 over 8 evenly spaced frames; silent = RMS < 1e-4 |
| `model` | `ckpt_sft` / `ckpt_epoch2` |

## Models
Both: PEFT LoRA r=8, α=16, dropout 0, on q/k/v/o/gate/up/down of all 28 Thinker LLM layers (392 tensors, FP32); audio and
vision encoders untouched; applied to Qwen2.5-Omni-7B Thinker and merged for inference. Adapter weights are not published here.
| adapter | what we know | sha256 of `adapter_model.safetensors` |
|---|---|---|
| `ckpt_sft` | adapter files only, no training state or config shipped | `ab16bfacb68d7574e088fd8e5edeb90e96687139067663ba33500778b4b053e2` |
| `ckpt_epoch2` | step 2,436 / 12,180 of a 10-epoch cosine run (lr 2e-5, warmup 3 %, lr still 1.85e-5 at this step), 8 GPU × bs 1 × GA 2; train loss 2.73 → 1.02 | `8b44d3149b881cf19bbb4ef9fb720a9c6dfde5c5c858de8aea9833c3c1ec661e` |

## Inference protocol (identical for both models)
- **Engine**: LLaMA-Factory 0.9.4 (`do_predict`, `predict_with_generate`), template `qwen2_omni`, bf16 + FlashAttention-2,
  batch 1 per GPU, 8 × A100-80GB per model. torch 2.7.1+cu126, transformers 4.54.0, peft 0.17.1, flash-attn 2.8.3.
- **Media settings** = the EgoAVU LoRA parameter table used for our EgoAVU training (`ckpt_epoch2`'s step count and LR schedule
  match that table; `ckpt_sft`'s training settings were not shipped). They differ from the official EgoAVU LoRA recipe
  (1 fps / ≤ 300 frames / 65,536 px):
  - each QA sees exactly its own `[start_time, end_time]` window (bench question timestamps are relative to that window);
  - video at 2 fps, capped at 64 frames (windows > 32 s — 3,773 of 3,976 — are uniformly subsampled to 64), ≤ 200,704 px/frame;
  - audio of the same window, 16 kHz, interleaved with the video in 2-s chunks (`use_audio_in_video`, TMRoPE). Qwen2.5-Omni's
    feature extractor keeps only the first 300 s of audio, so for the 822 windows longer than 300 s (max 347.6 s) the tail is
    video-only (`audio_capped_at_300s`; some of those questions ask about sounds after 300 s). Same behaviour as in training;
  - media read from 2 fps H.264 + FLAC proxies of the source Ego4D videos (equivalent input at these settings).
- **Prompt**: system `You are a helpful assistant.` (template default); user turn = the interleaved audio-visual tokens followed by
  the bench question verbatim; no added instruction. A long window is typically 8,064 video + 7,500 audio tokens.
- **Decoding**: greedy (`do_sample=false`, 1 beam), `max_new_tokens=1024` (longest gold answer: 483 tokens), `repetition_penalty=1.0`.
- **Join check**: every prediction was joined back to its bench row by order and verified — the pipeline's echoed label equals the
  gold answer and its prompt ends with the question, for 3,976/3,976 rows of both models. An independent re-check against the CSV
  found 0 mismatches in every field.

## Things the judges should know (lists of `bench_idx` in `notes/bench_window_notes.json`)
- **Black / silent input in the Ego4D source.** 106 windows (13 videos) are fully black, 90 of them also silent; 35 more windows
  (5 videos) are partly black — one of those videos (`ffd0a9b6`) is just a very dark scene. We checked all 18 videos against the
  raw Ego4D `full_scale` files: they are black / silent at the same spans, so this is the source data, not our preprocessing.
  The models were given this input as is.
- **Very short windows.** 24 windows are < 5 s (down to 0.125 s) while their gold answers (mostly dense narration) describe many
  10-s segments; 2 of them (`bench_idx` 1749, 2695) are < 1 s and had no audio.
- **Degenerate outputs.** All 18 (`ckpt_sft`) and 5 (`ckpt_epoch2`) outputs that hit the 1,024-token cap are repetition loops
  (listed in `stats/`); `ckpt_sft` also switches into Chinese in 6 outputs. Outputs are not post-processed.
- Outputs are much shorter than the gold answers in the narration categories (mean output vs gold tokens in `stats/`).

## Reproduction
`tools/build_bench_eval.py` (bench-ordered set + join assertions) → `tools/run_bench_infer.sh TAG ADAPTER DATASET 8` (restores the
env, writes the config with `tools/make_config.py`, runs predict, then `tools/collect_predictions.py`) → `tools/window_media_flags.py`
→ `tools/assemble_release.py`. `configs/` holds the exact predict configs; `submit_clusterx.sh` shows the cluster submission
(`ckpt_sft` ran as a P-cluster job, `ckpt_epoch2` on an 8-GPU dev box, both from the same env tarball).

## v3 (2026-09-24): the three EgoAVU LoRAs (ours)
`predictions/{r100k,r20k8g,r20k32g}.json`, `stats/`, per-shard configs in `configs/`, judge outputs in
`results/judge_items/<tag>.jsonl`, `results/judge_scores_egoavu_lora.csv`, captioning metrics `results/caption_*_egoavu_lora.csv`,
closed-ended items `results/closed_ended/<tag>_closed_items.jsonl`; `results/tables.md` / `summary.json` now cover all six models.
Same protocol as above, run on the H cluster: the bench was split into 8 interleaved shards (bench_idx % 8), two bs=1 predict
processes per H200, merged back in bench order (`tools/merge_bench_shards.py`) and re-verified with `collect_predictions.py`.
Official judge Qwen3-235B-A22B-Instruct-2507 (HF snapshot 56e16a6…, vLLM TP4 on H200), 11,928 / 11,928 items parsed.
