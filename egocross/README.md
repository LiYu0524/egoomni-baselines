# EgoCross (closed set) — untuned Qwen2.5-Omni-7B, the three EgoAVU LoRAs and the two colleague LoRAs

EgoCross: cross-domain egocentric video QA (arXiv 2508.10729), 957 four-way MCQs in 4 domains (Surgery 283, Industry 245,
XSports 246, Animal 183). **The test answers are hidden (Codabench leaderboard), so this directory holds only each model's
predictions in the Codabench submission format (`results/<tag>/predictions.json`, `{question_id: letter}`) and aggregate scores
(`results/<tag>/scores.json`) — no per-question gold answers.**

## Protocol (the benchmark's own `EgoCross/eval_os_closedset.py`)
- Input: the released frame list of each question as a video; sampling fps 0.5, except CholecTrack20 `.png` frames at 1.0
  (the eval script's rule); max 360×480 px per frame; no audio.
- Prompt: verbatim from `eval_os_closedset.py` (question + options + original/sampling fps sentence + JSON answer format);
  greedy decoding, ≤512 new tokens; `qwen2_omni` template, LLaMA-Factory predict.
- Parse: the script's `parse_model_response` (json.loads, else the first `{...}` block → `prediction`); scoring: the official
  Codabench `scoring.py` rules (normalize, letter-or-text match, per domain). A lenient parse (bare option letter) is also reported.
- 41 questions list frame files that are absent from the released testbed (also absent from its ModelScope file list; 38 of them
  EgoSurgery); each missing frame is replaced by its nearest existing frame so frame count and timestamps stay unchanged.
  `scores.json` reports all 957 questions (`official_parse`) and the 916 with complete frames (`official_parse_complete_frames`).

## Results (official parse, accuracy %)
| model | overall | Surgery | Industry | XSports | Animal | complete-frame subset | vs base (exact McNemar) |
|---|---|---|---|---|---|---|---|
| base (Qwen2.5-Omni-7B) | 45.25 | 43.1 | 46.5 | 44.3 | 48.1 | 45.63 | — |
| r100k | 43.89 | 40.3 | 45.3 | 45.5 | 45.4 | 44.00 | −1.36, p = 0.40 |
| r20k-8gpu | 45.66 | 41.7 | 43.7 | 50.0 | 48.6 | 45.96 | +0.42, p = 0.82 |
| r20k-32gpu | 45.14 | 39.2 | 44.9 | 50.4 | 47.5 | 45.52 | −0.10, p = 1.00 |
| ckpt_sft (colleague LoRA) | 44.41 | 42.8 | 44.1 | 45.9 | 45.4 | 44.87 | −0.84, p = 0.54 |
| ckpt_epoch2 (colleague LoRA) | 43.57 | 41.7 | 46.1 | 42.7 | 44.3 | 43.78 | −1.67, p = 0.23 |

For reference, earlier submissions scored with the same rules: Qwen3-VL-4B 45.14, Qwen3-VL-4B full-SFT 46.08,
Qwen3.5-397B-A17B 48.48 (931 answered).

## Files
`build_data.py` (LLaMA-Factory datasets from the testbed), `make_config.py`, `collect.py` (parse + score); job wrapper
`../h_cluster/egocross_h.sh` (one H200, two predict processes).
