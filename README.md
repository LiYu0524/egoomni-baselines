# egoOmni baselines

Inference outputs and the evaluation harness for four audio-visual LLM baselines on the **egoOmni** test set
(`grooLegend/egoOmni` on Hugging Face: 3765 items = 3040 single-turn EN + 725 multi-turn ZH, 5040 answer turns, 1905 clips).

| model | weights | predictions | judge |
|---|---|---|---|
| VideoLLaMA2.1-7B-AV | `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV` (repo branch `audio_visual`) | `eval/preds/videollama2_7b_av/` | ✅ `eval/judgments/`, `eval/results/videollama2_7b_av/` |
| video-SALMONN 2+ 7B | `tsinghua-ee/video-SALMONN2_plus_7B_full` | `eval/preds/salmonn2plus_7b/` | pending |
| video-SALMONN 2+ 72B | `tsinghua-ee/video-SALMONN2_plus_72B_full` | `eval/preds/salmonn2plus_72b/` | pending |
| MiniCPM-o 2.6 (8B) | `openbmb/MiniCPM-o-2_6` | `eval/preds/minicpmo_2_6_8b/` | pending |

All four prediction sets are complete: 7040 rows each (5040 `gold` + 2000 `self` protocol rows), 0 inference errors.

## Protocol (details in [`eval/README.md`](eval/README.md))
- Full clip as input; audio fed iff the clip has an audio stream (41 % of clips have none). Each model at its official settings
  (VideoLLaMA2.1: 16 frames + BEATs; SALMONN 2+: 768 frames / 61250 px / 0.1 s, the paper's eval setting; MiniCPM-o 2.6: official omni
  mode, 1-second units of frame + audio, clips > 128 s uniformly subsampled to 128 units). Greedy, ≤256 new tokens.
- Prompts: open questions get a one-line "answer briefly" suffix (EN/ZH); MCQ (79 items) = options + "answer with the letter".
- Multi-turn: `gold` = round k conditioned on dataset answers of rounds <k (headline); `self` = conditioned on the model's own answers.
- Scoring: MCQ by letter match; open answers by an LLM judge (`eval/egoomni_eval/judge.py`; local Qwen3-32B via vLLM by default,
  any OpenAI-compatible endpoint via `--backend openai`). `eval/egoomni_eval/score.py` produces the tables.

## Prediction row schema (`eval/preds/<tag>/shard*.jsonl`)
`key` (`<item_id>#r<turn>`), `item_id`, `turn_idx`, `n_turns`, `protocol`, `clip_rel`, `has_audio`, `duration`, `lang`, `fmt`,
`question`, `messages` (the exact chat history sent), `gold`, `options`, `correct_options`, `meta` (category, subcategory, track,
min_modalities, …), `pred`, `error`, `modality_used`, `n_input_tokens`, `latency_s`, `prep_wait_s`, `ts`.

## First result — VideoLLaMA2.1-7B-AV (judge: Qwen3-32B)
overall **20.65 %** (gold) / 18.95 % (self) · single-turn 21.1 % · multi-turn round 20.0 % (gold) / 15.7 % (self) ·
all-rounds-correct 2.1 % · MCQ 68.4 % · open 19.9 %. Breakdowns in `eval/results/videollama2_7b_av/gold.md`.

## Reproducing
`SETUP.md` documents the machine setup (model downloads via `scripts/download_models.sh`, conda envs via `scripts/build_env_*.sh`,
checkpoint validation via `scripts/salmonn_validate_*.py`). Then:
```bash
cd eval && python prepare_clips.py                      # ffprobe → clips_meta.json (already included)
./launch_infer.sh videollama2|salmonn7b|salmonn72b|minicpmo   # 8 GPUs; 72B uses DeepSpeed ZeRO-3 data-parallel
./launch_judge.sh <tag> && python -m egoomni_eval.score --tag <tag> --protocol gold|self
```
Known deviations from the upstream inference code (GPU-side frame preprocessing for SALMONN, multi-turn prompt cut, ZeRO-3 audio/no-audio
lockstep phases) are documented and quantified in `eval/README.md`.
