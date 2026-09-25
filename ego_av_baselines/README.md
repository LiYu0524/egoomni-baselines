# EgoSound / EgoToM / EgoTempo — five baseline models (2026-09-26)

This folder fills the empty cells of the baseline table (EgoSound | EgoToM | EgoTempo × EgoGPT-7B, video-SALMONN2+-7B,
MiniCPM-o 2.6-8B, Qwen2-VL-7B, LLaVA-OneVision-Qwen2-7B). Everything was run on the H cluster (H200, one GPU per job); code,
per-question predictions, judge verdicts and scores are here.

## Main table

★ = **new run by us** (this folder) · † = published number, copied unchanged

| Model | EgoSound | EgoToM | EgoTempo |
|---|---|---|---|
| EgoGPT-7B | 34.30 † | **63.88** ★ | **16.80** ★ |
| video-SALMONN2+-7B | 36.00 † | **61.05** ★ | **23.60** ★ |
| MiniCPM-o 2.6-8B | 40.40 † | **57.51** ★ | **25.20** ★ |
| Qwen2-VL-7B | **31.72** ★ | **61.19** ★ | 26.10 † |
| LLaVA-OneVision-Qwen2-7B | **33.60** ★ | **63.31** ★ | 23.30 † |

† EgoSound: Zhu et al., 2026 (EgoSound, CVPR 2026). EgoTempo: Plizzari et al., 2025 (EgoTempo, CVPR 2025).

- **EgoSound** — accuracy (%) over all 7,315 QAs (2,346 Ego4D + 4,969 EgoBlind), judged by **gpt-5** with the official prompt.
  Qwen2-VL and LLaVA-OneVision take no audio, so they answer from the video frames alone.
  Per subset (Ego4D / EgoBlind): Qwen2-VL 24.55 / 35.10, LLaVA-OneVision 30.22 / 35.20.
- **EgoToM** — multiple-choice accuracy (%) on the **paper question set** (`egotom_paper/`, 706 questions: 237 goal, 202 belief,
  267 actions) with the **last-30-s video context**. All three context conditions and the full 1,039-question release are
  [below](#egotom--every-context-condition-and-question-type).
- **EgoTempo** — accuracy (%) over all 500 open-ended QAs judged with the official prompt by **gemini-2.5-flash-lite**; the
  official judge (gemini-1.5-flash) was not available, see [the calibration](#egotempo-judge-calibration). With gemini-2.5-flash
  as the judge the three new cells read 14.60 / 21.60 / 20.60.

**Mixing caveat.** The ★ and † cells of one column come from different pipelines. For EgoTempo the judge differs (ours:
gemini-2.5-flash-lite; published: gemini-1.5-flash); for EgoSound the judge is the same (gpt-5) but the published models were run
with their own audio-visual inputs while our two are video-only. Our own re-runs of the two published EgoTempo cells are
Qwen2-VL 23.00 (published 26.10) and LLaVA-OneVision 24.80 (published 23.30).

## Does it strictly follow each benchmark's official protocol?

**Not entirely.** Everything a benchmark defines was followed except the items marked ⚠️. Most ⚠️ items are model-side settings the
benchmark does not define for these models, plus one unavoidable change: the EgoTempo judge.

✅ = identical to the official release · ⚠️ = deviation or a choice the benchmark leaves open

### EgoSound (our cells: Qwen2-VL, LLaVA-OneVision)

| item | official | ours | |
|---|---|---|---|
| data | HF `grooLegend/EgoSound` | same repo, rev `06a8a9e`, all 7,315 QAs, sha256-verified | ✅ |
| prompt | the question text only (official inference scripts) | same | ✅ |
| model inference | scripts only for EgoGPT, MiniCPM-o, Qwen2.5-Omni, VideoLLaMA2, Qwen3-Omni | none exist for Qwen2-VL / LLaVA-OV → 32 uniform frames, video only (no audio input in these models), greedy, ≤ 1,024 new tokens | ⚠️ defined by us |
| judge | `qa_eval_gpt.py`: gpt-5, system + user prompt, `max_completion_tokens` 800, accuracy = `binary_pred == 'correct'` | identical (prompt copied verbatim), served through an OpenAI-compatible gateway | ✅ |
| reply parsing | `ast.literal_eval` | same; 0 replies needed a fallback | ✅ |

### EgoToM (all five cells)

| item | official | ours | |
|---|---|---|---|
| questions | `facebookresearch/EgoToM`: `egotom/` (1,039) and `egotom_paper/` (706) | both, all questions | ✅ |
| video contexts | `code/generate_video_context.py`: moviepy 1.0.3 `ffmpeg_extract_subclip` + `ffmpeg_resize` to [480, 720], from the original Ego4D videos | exactly those calls (moviepy 1.0.3), Ego4D v2.1 `full_scale` (291 videos from our size-verified mirror of the official files, 332 straight from Ego4D's S3), all three conditions; 0 clamped windows, 0 fallbacks | ✅ |
| prompts | `all_prompts.json` (system + user) | same | ✅ |
| frames (Qwen2-VL, LLaVA-OV) | `utils.select_video_frames('uniform24')` (end-aligned) | same; images go through each model's own processor (the config's `video_size: [224, 224]` is the CogVLM2 example) | ✅ / ⚠️ resolution |
| frames (EgoGPT, SALMONN2+, MiniCPM-o) | the repo has no loaders for these models | each model's native audio-visual input, same as in EgoSound's official scripts (see [Model settings](#model-settings)) | ⚠️ |
| decoding | temperature 0 | greedy, ≤ 1,024 new tokens (MiniCPM-o reasons before answering; a 128-token cap cut it off) | ✅ |
| scoring | not released | letter read from the requested `Answer <q>: <option>)` format, else a unique verbatim option; unparsed: EgoGPT 2, SALMONN2+ 3, others 0 of 3,117 | ⚠️ ours |
| reported setting | paper reports several conditions | table uses paper set + last30sec; the full grid is below | ⚠️ choice |

### EgoTempo (our cells: EgoGPT, SALMONN2+, MiniCPM-o)

| item | official | ours | |
|---|---|---|---|
| questions | `google-research-datasets/egotempo` `egotempo_openQA.json` | same, all 500 QAs / 367 clips | ✅ |
| source videos | Ego4D `video_540ss` | same for 366 clips; one `540ss` object is forbidden on Ego4D's S3 → that clip from `full_scale`, scaled to 540p | ✅ / ⚠️ 1 clip |
| clip cutting | `ffmpeg -i src -ss S -to E -c copy` (stream copy: starts at the first packet ≥ S, not on a keyframe) | exact [S, E] window re-encoded (libx264 crf 18, AAC 128k) — decodable from the first frame | ⚠️ |
| prompt | the notebook's question wrapper | copied verbatim | ✅ |
| frames | not specified for open models | 32 uniform frames (Qwen2-VL, LLaVA-OV); native audio-visual input for the others | ⚠️ |
| judge model | gemini-1.5-flash | not offered by the gateway → **gemini-2.5-flash-lite**, chosen by calibration below | ⚠️ unavoidable |
| judge prompt | `create_prompt()` | copied verbatim | ✅ |
| reply parsing | first `{…}` → `ast.literal_eval`; unparseable replies are **dropped** | same, but unparseable replies (7–27 per model, usually an apostrophe inside `reason`) are read with a regex on `pred` / `score` instead of dropped | ⚠️ ≤ 0.9 pt |

EgoTempo accuracy under the official drop-unparseable rule (gemini-2.5-flash-lite): EgoGPT 16.84 (n = 493), SALMONN2+ 24.52
(473), MiniCPM-o 25.77 (485), Qwen2-VL 23.36 (488), LLaVA-OneVision 25.67 (483).

### Libraries

| model | pinned by its authors | used here |
|---|---|---|
| MiniCPM-o 2.6 | torch 2.3.1, transformers 4.44.2 | torch 2.7.1, **transformers 4.44.2** |
| EgoGPT | torch 2.1.2, transformers 4.45.2, deepspeed | torch 2.7.1, **transformers 4.45.2**; deepspeed replaced by a no-op `GatheredParameters` stub ([`eval/deepspeed_stub/`](eval/deepspeed_stub/)) — the real call is a no-op too for non-partitioned single-GPU weights |
| video-SALMONN2+ | torch 2.7.1, transformers 4.51.3, flash-attn 2.7.4.post1 | **torch 2.7.1, transformers 4.51.3**, flash-attn 2.8.3 (flash-attention 2 as in its `inference.py`) |
| Qwen2-VL, LLaVA-OV | — | torch 2.7.1, transformers 4.56.2 (HF-native classes) |

## EgoTempo judge calibration

The two cells with published EgoTempo numbers, re-run with our pipeline and judged by every usable candidate on the gateway
(gemini-2.0-flash-lite is listed but has no serving channel):

| judge | Qwen2-VL (pub. 26.10) | LLaVA-OV (pub. 23.30) | mean abs. gap |
|---|---|---|---|
| **gemini-2.5-flash-lite** (used) | 23.00 | 24.80 | **2.3** |
| gemini-2.5-flash, no reasoning | 21.69 | 22.18 | 2.8 |
| gemini-2.5-flash | 19.80 | 21.80 | 3.9 |
| gpt-4o-mini | 18.40 | 16.60 | 7.2 |

No judge reproduces both published cells, because inference settings (frames, resolution) are also unpublished. New cells under both
Gemini judges:

| model | gemini-2.5-flash-lite | gemini-2.5-flash |
|---|---|---|
| EgoGPT-7B | 16.80 | 14.60 |
| video-SALMONN2+-7B | 23.60 | 21.60 |
| MiniCPM-o 2.6-8B | 25.20 | 20.60 |
| Qwen2-VL-7B (calibration) | 23.00 | 19.80 |
| LLaVA-OneVision-Qwen2-7B (calibration) | 24.80 | 21.80 |

## EgoToM — every context condition and question type

Accuracy (%) as goal / belief / actions → **overall** (micro-average over questions).

**Paper set** (`egotom_paper/`, 706 questions: 237 / 202 / 267)

| Model | fullcontext | last30sec | last5sec |
|---|---|---|---|
| EgoGPT-7B | 93.7/48.0/38.2 → **59.63** | 96.2/47.0/47.9 → **63.88** | 94.5/46.0/41.6 → **60.62** |
| video-SALMONN2+-7B | 87.8/42.6/30.7 → **53.26** | 92.0/51.0/41.2 → **61.05** | 90.7/49.5/48.7 → **63.03** |
| MiniCPM-o 2.6-8B | 90.7/38.1/40.5 → **56.66** | 88.6/40.6/42.7 → **57.51** | 86.1/43.1/42.7 → **57.37** |
| Qwen2-VL-7B | 92.0/41.6/34.1 → **55.67** | 92.0/50.5/42.0 → **61.19** | 93.7/54.5/49.8 → **65.86** |
| LLaVA-OneVision-Qwen2-7B | 94.5/45.0/41.2 → **60.20** | 95.4/46.0/47.9 → **63.31** | 93.2/50.5/54.7 → **66.43** |

**Full release** (`egotom/`, 1,039 questions: 351 / 334 / 354)

| Model | fullcontext | last30sec | last5sec |
|---|---|---|---|
| EgoGPT-7B | 93.7/45.2/38.1 → **59.19** | 96.6/48.8/45.8 → **63.91** | 94.6/45.8/39.5 → **60.15** |
| video-SALMONN2+-7B | 87.8/42.2/32.2 → **54.19** | 91.7/49.7/41.2 → **61.02** | 90.9/49.4/48.6 → **63.14** |
| MiniCPM-o 2.6-8B | 89.7/39.2/38.1 → **55.92** | 88.0/41.3/43.2 → **57.75** | 85.2/45.2/41.8 → **57.56** |
| Qwen2-VL-7B | 91.5/40.4/30.8 → **54.38** | 92.0/48.8/39.3 → **60.15** | 93.5/54.5/47.5 → **65.26** |
| LLaVA-OneVision-Qwen2-7B | 93.7/42.5/38.4 → **58.42** | 94.3/47.0/48.3 → **63.43** | 93.2/48.8/52.3 → **64.97** |

As in the EgoToM paper, goal questions are near ceiling and belief / action questions stay near 40–55 %.

## Model settings

| model | weights | video input | audio | notes |
|---|---|---|---|---|
| EgoGPT-7B | `lmms-lab/EgoGPT-7b-EgoIT-EgoLife` (Whisper large-v3 ships with it) + `google/siglip-so400m-patch14-384` | 1 fps, ≤ 16 uniform frames | first 30 s → Whisper 128-bin log-mel (EgoSound's `.wav`, else the clip's track) | EgoSound's `egogpt_av_inference.py` |
| video-SALMONN2+-7B | `tsinghua-ee/video-SALMONN2_plus_7B_full` (rev `aab28bd`) | interval 0.1 s, 16–768 frames, ≤ 61,250 px/frame | clip track (Whisper features) | the repo's `inference.py` / README eval setting, flash-attention 2 |
| MiniCPM-o 2.6-8B | `openbmb/MiniCPM-o-2_6` | 1 frame + 1 s audio per unit (2 s units above 240 s) | yes | EgoSound's `minicpm_av_infer.py`, greedy instead of its T = 0.5 sampling |
| Qwen2-VL-7B | `Qwen/Qwen2-VL-7B-Instruct` | EgoToM uniform24 (end-aligned) / 32 uniform elsewhere | — | HF processor defaults |
| LLaVA-OV-7B | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | same | — | HF processor defaults |

All models: greedy decoding, ≤ 1,024 new tokens, one GPU per process, bf16 (EgoGPT fp16 as in its script).

## Reproduce (H cluster)

Media live in object storage (not in this repo): `s3://safevlagent/liyu/benchmarks/{egosound,egotom,egotempo}/`.

1. **Clips** — [`clips/build_tasks.py`](clips/build_tasks.py) (openvla: presigned source URLs, per-bucket region) →
   [`clips/cut_clips.py`](clips/cut_clips.py) `egotempo` / `egotom-stage1` on openvla (the only host with internet; ffmpeg reads
   through [`clips/rangeproxy.py`](clips/rangeproxy.py)) and `egotom` as a CPU rjob ([`clips/run_egotom_rjob.sh`](clips/run_egotom_rjob.sh)).
2. **Requests** — [`eval/build_requests.py`](eval/build_requests.py) (EgoToM 3,117 = 1,039 × 3 conditions, EgoTempo 500, EgoSound 7,315).
3. **Inference** — [`eval/run_infer.sh`](eval/run_infer.sh) `MODEL BENCH NUM_SHARDS FIRST_SHARD PROCS` in 1-GPU rjobs (stages media
   to the pod's overlay disk with [`eval/stage_media.py`](eval/stage_media.py)); adapters `eval/infer_*.py`; environments
   [`eval/build_envs.sh`](eval/build_envs.sh); retries until complete: [`eval/chain_complete.sh`](eval/chain_complete.sh).
4. **Scoring** — [`eval/score_egotom.py`](eval/score_egotom.py); [`eval/judge_openended.py`](eval/judge_openended.py) `egosound|egotempo MODEL
   [--judge NAME]` (needs an OpenAI-compatible endpoint in `/root/.gpt_judge.env`, never committed); [`eval/summarize.py`](eval/summarize.py).

## Files

| path | content |
|---|---|
| [`results/SUMMARY.md`](results/SUMMARY.md), [`results/summary.json`](results/summary.json) | the tables above |
| `results/<model>/egotom.json` | accuracy per split × condition × question type, unparsed counts |
| `results/<model>/<bench>__<judge>.json` | judged accuracy / mean score overall, per subset and question type |
| `preds/<model>/<bench>.jsonl` | one line per question: `id`, `pred` (+ frames / audio flags) |
| `judgments/<bench>/<model>__<judge>.jsonl` | judge verdict, score and raw reply per question |
| `clips/`, `eval/` | all code used |
