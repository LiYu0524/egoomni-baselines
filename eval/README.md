# egoOmni evaluation harness (`eval/`)

Evaluates audio-visual LLMs on egoOmni test (`/ai4good1-shared/liyu/egoOmni/test/qa.json`: 3765 items = 3040 single-turn EN
+ 725 multi-turn ZH; 5040 answer turns; 1905 clips, 782 without an audio stream). Built 2026-09-19.

## Protocol
- **Inputs**: full clip, no evidence-window cropping. Audio is fed iff the clip has an audio stream (`modality_used` recorded per row).
  Model-specific official settings: VideoLLaMA2.1-AV 16 frames + BEATs; video-SALMONN 2+ `768 frames / 61250 px / 0.1 s` (paper eval setting).
  Greedy decoding, `max_new_tokens 256`.
- **Prompts** (`egoomni_eval/prompts.py`, `PROMPT_VERSION v1`, stored verbatim in every row): open EN `…\nAnswer briefly in one or two sentences.`,
  open ZH `…\n请用一两句话简要回答。`, MCQ = question + `Options:` + `Answer with the option's letter from the given choices directly.`
- **Multi-turn** (725 items, 2–4 rounds): `gold` = round k conditioned on the dataset answers of rounds <k (headline);
  `self` = conditioned on the model's own earlier answers (round 1 identical to gold, copied). Both are produced in one pass.
- **Scoring**: MCQ (79) = parsed letter vs `correct_options` (unparseable ⇒ wrong, counted); open = LLM judge binary verdict
  (`egoomni_eval/judge.py`, `JUDGE_VERSION v1`; default local Qwen3-32B via vLLM, temperature 0, JSON output; `--backend openai` for any
  OpenAI-compatible endpoint via `JUDGE_BASE_URL/JUDGE_API_KEY`). Inference errors are counted as wrong and reported separately.
- **Reported** (`egoomni_eval/score.py` → `results/<tag>/{gold,self}.{json,md}`): overall / single-turn / multi-turn round accuracy,
  all-rounds-correct rate, and breakdowns by format, track, category, subcategory, minimum modality, clip audio, modality used, language, round, clip duration.

## Deviations from the repos' own inference code (documented on purpose)
- **SALMONN frame resize/normalize on GPU** (`processor.preprocess(device="cuda")`, chunked 64 frames — bit-identical to whole-video):
  the repo's CPU uint8 path costs 40–80 s per clip vs 0.5 s. Frame sampling and token counts are identical; pixel values differ by
  mean 9.6e-4 (max 0.09; the CPU uint8 kernel is itself quantization-noisy — it differs from float math by up to 0.19). On 16 items,
  12/16 predictions were identical and 4 were paraphrases. `--no_gpu_preprocess` restores the exact repo path; stage 3 runs it on dev300
  and writes `results/preprocessing_deviation_dev300.json`.
- **Multi-turn prompt cut**: the repo's `run_test` cut (`sum(labels==IGNORE)`) is single-turn only; we cut at the last `<|im_start|>assistant\n`
  (identical for single-turn).
- **72B**: DeepSpeed ZeRO-3 data-parallel over 8 ranks (the mechanism of the repo's `test_8.sh`). Every param all-gather is a collective,
  so all ranks must run the same modules: clips with/without audio are processed in two lockstep phases, each padded with dummy generations.

## Layout / commands
```
egoomni_eval/{data,prompts,run_infer,judge,score}.py  models/{videollama2,salmonn2plus}.py   clips_meta.json (ffprobe of all clips)
preds/<tag>/shard*.jsonl   judgments/<tag>.jsonl   results/<tag>/{gold,self}.{json,md}   subsets/dev300.json   logs/
./launch_infer.sh videollama2|salmonn7b|salmonn72b [--subset subsets/dev300.json] [--retry_errors]   # 8 workers / torchrun; blocks
GPU_UTIL=0.2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./launch_judge.sh <tag> [...]      # co-run with inference; 0.85 on 2 GPUs when alone
$EGO_ENVS/judge/bin/python -m egoomni_eval.score --tag <tag> --protocol gold|self
run_stage2.sh / run_stage3.sh   # the chained pipeline used for the first full run (see logs/stage*.log)
```
Tags: `videollama2_7b_av`, `salmonn2plus_7b`, `salmonn2plus_72b`. Workers are resumable (skip rows already in their shard file).

## Adding a model (the 5 Omni baselines)
Subclass `egoomni_eval/models/base.py::ModelAdapter` in its own conda env: `load()`, `prepare_media(clip, has_audio)` (CPU-heavy; runs on
prefetch threads, use `self.gpu_lock` for GPU work), `generate(media, messages)` → `{"pred", "modality_used", "n_input_tokens"}`;
register in `models/__init__.py`, add an entry to `launch_infer.sh`. Nothing else changes.

## Throughput (8×A100-80GB, first full run)
VideoLLaMA2.1-AV: 2.7 s/gen, 25 min total. SALMONN 2+ 7B: ~8 s/gen (CPU decode fully overlapped), ~1.5 h. SALMONN 2+ 72B ZeRO-3: ~130 s/gen/rank ⇒ ~16 s/gen effective.
Judge: ~20 verdicts/s (TP=8 @ 20 % memory, co-resident).
