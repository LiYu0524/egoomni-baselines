# egoOmni baselines (+ EgoAVU-Bench, EgoCross, EgoSchema, EgoTaskQA)

Inference outputs and evaluation code for five benchmarks:

| benchmark | size | models | where | status |
|---|---|---|---|---|
| **egoOmni** — final test set (`test.qa.jsonl`, 3,882 QAs = 3,765 original + 117 restored) | 3,882 QAs | VideoLLaMA2.1-AV-7B, video-SALMONN 2+ 7B / 72B, MiniCPM-o 2.6, Gemini 3.8 Flash, untuned Qwen2.5-Omni-7B, **EgoAVU LoRAs (ours): r100k, r20k-8gpu, r20k-32gpu**, colleague checkpoints `ckpt_sft`, `ckpt_epoch2`, `ckpt_fft_epoch2` | [`eval/`](eval/README.md) | predictions for **3,877 / 3,882** QAs on all models (5 restored QAs have no video); **all 12 models judged** (Qwen3-32B) |
| **EgoAVU-Bench** | 3,976 QAs, 500 videos | the seven Qwen2.5-Omni-7B models below | [`egoavu_bench/`](egoavu_bench/README.md) | official judge (Qwen3-235B-A22B-Instruct-2507) + official captioning metrics: v2 base control, v3 our LoRAs, 2026-09-25 full FT + protocol re-test ([`results/`](egoavu_bench/results/README.md)) |
| **EgoCross** (closed set) | 957 MCQs, 4 domains | the seven Qwen2.5-Omni-7B models | [`egocross/`](egocross/README.md) | done; predictions (Codabench format) + aggregate scores only — the test answers are hidden |
| **EgoSchema** (public Subset) | 500 five-way MCQs | the seven Qwen2.5-Omni-7B models | [`egoschema/`](egoschema/README.md) | done (full 5,031-question set not run: answers are server-side) |
| **EgoTaskQA** (test: direct + indirect) | 18,936 short-answer QAs | the seven Qwen2.5-Omni-7B models | [`egotaskqa/`](egotaskqa/README.md) | done; exact match + Qwen3-32B judge |

**The seven Qwen2.5-Omni-7B models:** the untuned base; our three EgoAVU LoRAs (r8/α16, 5 epochs on EgoAVU window groups:
**r100k** = 100k-row subset on 8 GPUs, **r20k-8gpu** / **r20k-32gpu** = 20k-row subset on 8 / 32 GPUs; adapters in private ModelScope
repos, not here); and three colleague checkpoints: `ckpt_epoch2` = ModelScope `groo_legend/ckpts` `qwen2.5omni7b/ckpt_lora_epoch2`
(byte-identical), `ckpt_fft_epoch2` = the full fine-tune in the same repo (thinker-only checkpoint, re-headered to the full-Omni
thinker view with byte-identical weights by [`h_cluster/convert_thinker_ckpt.py`](h_cluster/convert_thinker_ckpt.py); it changed every
module incl. both encoders), `ckpt_sft` = a colleague LoRA that is not in that repo.

Every problem hit and how it was handled: [`docs/ISSUES_AND_HANDLING.md`](docs/ISSUES_AND_HANDLING.md).

## Results — seven Qwen2.5-Omni-7B models on five ego benchmarks (2026-09-25)

All runs on the H cluster (H200), greedy decoding; scripts in [`h_cluster/`](h_cluster/README.md). Table 1 is the protocol every model
got. Table 2 re-tests the colleague's ModelScope checkpoints — with the base as the same-protocol reference — under the colleague's own
EgoToM inference kit; Table 3 compares the two protocols question by question.

### Table 1 — original protocol

LLaMA-Factory predict with each benchmark's prompt and scoring; EgoAVU training media recipe (2 fps, ≤ 64 frames, ≤ 200,704 px per
frame, audio interleaved with the video when the clip has sound). Best per column in bold.

| model | EAB SSA S | EAB AVSN S | EAB AVSN M | EAB TR | EAB AVH | egoOmni gold / self | EgoCross | EgoSchema | EgoTaskQA direct EM / judge | EgoTaskQA indirect EM / judge |
|---|---|---|---|---|---|---|---|---|---|---|
| base (untuned Qwen2.5-Omni-7B) | 1.50 | 1.92 | 10.47 | 44.6 | 23.6 | 36.80 / 35.01 | 45.25 | **65.2** | **29.96 / 34.52** | **34.62** / 38.17 |
| r100k (ours) | **2.65** | 2.01 | **20.59** | 44.2 | **98.4** | 28.69 / 26.60 | 43.89 | 58.0 | 22.16 / 29.50 | 27.86 / 33.28 |
| r20k-8gpu (ours) | **2.65** | 2.05 | 19.33 | 42.0 | 97.5 | 30.12 / 28.77 | **45.66** | 64.0 | 25.06 / 30.88 | 29.77 / 34.17 |
| r20k-32gpu (ours) | 2.62 | **2.06** | 19.57 | **47.2** | 97.1 | 29.56 / 27.94 | 45.14 | 60.4 | 24.50 / 30.45 | 29.35 / 34.12 |
| ckpt_sft (colleague LoRA) | 1.53 | 1.68 | 4.39 | 43.6 | 13.0 | **52.30 / 50.41** | 44.41 | 56.4 | 25.65 / 33.31 | 31.02 / 36.81 |
| ckpt_epoch2 = ckpt_lora_epoch2 (colleague LoRA) | 1.56 | 1.67 | 5.11 | 45.0 | 36.1 | 45.58 / 43.68 | 43.57 | 54.2 | 24.52 / 33.71 | 31.04 / **38.65** |
| ckpt_fft_epoch2 (colleague full FT) | 1.56 | 1.69 | 5.46 | 44.8 | 22.5 | 48.36 / 46.70 | 41.07 | 57.6 | 23.10 / 32.62 | 29.07 / 36.41 |

**Metrics**
- **EAB** (EgoAVU-Bench, official protocol): S = official LLM-judge score 1–5 (Qwen3-235B-A22B-Instruct-2507) for SSA (sound source
  association) and AVSN (audio-visual segment narration); M = METEOR ×100 on AVSN; TR = temporal-reasoning MCQ accuracy; AVH = accuracy
  on the 304 yes/no hallucination probes (macro over action / object / sound; 290 have the gold answer "No", so always-"No" scores 95.4).
- **egoOmni**: accuracy (%) over all 5,176 gold rows (MCQ by letter, open answers by a Qwen3-32B judge); *gold* = each turn conditioned
  on the dataset's answers to the earlier turns, *self* = on the model's own earlier answers.
- **EgoCross**: official Codabench closed-set accuracy. **EgoSchema**: public 500-question Subset, lmms-eval prompt and answer parse
  (the released videos have no audio track).
- **EgoTaskQA**: two test splits — *direct* (8,783 QAs: questions name the objects / actions) and *indirect* (10,153 QAs: they are
  referred to through other events, e.g. "the object used before …"). **EM** = normalized exact match (lower-cased, punctuation and
  articles removed; yes/no questions compare the first word only). **judge** = EM plus, for every non-yes/no answer EM rejected,
  Qwen3-32B (vLLM, thinking off, greedy) deciding whether it means the same as the reference (paraphrases and synonyms accepted, a
  different object / action / state / attribute rejected) — so judge ≥ EM. Generative models phrase short answers freely
  ("chopping board" vs "cutting board"), which EM alone under-counts.

### Table 2 — the colleague's EgoToM-kit protocol (colleague checkpoints + base)

The colleague's EgoToM inference kit, copied step by step in [`h_cluster/nat_infer.py`](h_cluster/nat_infer.py) /
[`h_cluster/nat_predict_lf.py`](h_cluster/nat_predict_lf.py): native transformers `Qwen2_5OmniForConditionalGeneration` (bf16,
FlashAttention-2, talker disabled), a LoRA merged into the weights with every adapter tensor verified, `qwen_omni_utils.process_mm_info`
with decord, video at **2 fps**, `min_pixels` 224×224, **`max_pixels` 156,800**, the clip's soundtrack interleaved as audio-in-video,
greedy, 256 new tokens, seed 42. Generalized from the kit's 30 s EgoToM clips: system prompt "You are a helpful assistant." (these
benchmarks ship none; it is the chat-template default and what Table 1 used); at most 360 frames (EgoAVU-Bench windows reach 348 s —
EgoSchema's full 3 min at 2 fps is 360 frames, 34.7k input tokens); EgoAVU-Bench keeps its 1,024-token budget (gold answers reach 483
tokens); audio from the 16 kHz FLAC extracted from the same clip (the EgoAVU-Bench proxies carry no audio track); 2 EgoAVU-Bench windows
shorter than 1 s widened to 1 s. Cell = kit result (Δ vs Table 1). egoOmni was not re-tested.

| model | EAB SSA S | EAB AVSN S | EAB AVSN M | EAB TR | EAB AVH | egoOmni | EgoCross | EgoSchema | EgoTaskQA direct EM / judge | EgoTaskQA indirect EM / judge |
|---|---|---|---|---|---|---|---|---|---|---|
| base (untuned Qwen2.5-Omni-7B) | 1.45 (−0.05) | 1.82 (−0.09) | 11.13 (+0.66) | 42.4 (−2.2) | 29.3 (+5.7) | — | 45.66 (+0.41) | 63.8 (−1.4) | 30.07 / 34.41 (+0.11 / −0.11) | 34.60 / 38.33 (−0.02 / +0.16) |
| ckpt_epoch2 = ckpt_lora_epoch2 (colleague LoRA) | 1.54 (−0.02) | 1.65 (−0.02) | 5.19 (+0.08) | 43.6 (−1.4) | 42.5 (+6.4) | — | 45.66 (+2.09) | 54.6 (+0.4) | 24.42 / 33.58 (−0.10 / −0.13) | 30.84 / 38.33 (−0.20 / −0.32) |
| ckpt_fft_epoch2 (colleague full FT) | 1.55 (−0.02) | 1.66 (−0.03) | 5.67 (+0.21) | 44.2 (−0.6) | 28.5 (+6.0) | — | 42.22 (+1.15) | 56.8 (−0.8) | 23.22 / 32.64 (+0.12 / +0.02) | 28.99 / 36.26 (−0.08 / −0.15) |

### Table 3 — kit vs original protocol, same model, paired per question (exact McNemar)

| model | benchmark | questions | only the original protocol correct | only the kit correct | p |
|---|---|---|---|---|---|
| base | EgoSchema | 500 | 37 | 30 | 0.46 |
| base | EgoCross | 957 | 41 | 45 | 0.75 |
| ckpt_epoch2 | EgoSchema | 500 | 33 | 35 | 0.90 |
| ckpt_epoch2 | EgoCross | 957 | 31 | 51 | **0.035** |
| ckpt_fft_epoch2 | EgoSchema | 500 | 33 | 29 | 0.70 |
| ckpt_fft_epoch2 | EgoCross | 957 | 42 | 53 | 0.31 |

### Takeaways
1. **The kit protocol changes almost nothing.** On EgoSchema, EgoCross and EgoTaskQA every shift is within ±1.5 points except
   `ckpt_epoch2` on EgoCross (+2.1, p = 0.035); on EgoAVU-Bench the judge scores move by ≤ 0.1 and AVH rises by ~6 points for every
   model (fewer "Yes" answers). The colleague checkpoints' EgoSchema deficit vs the base (−7 to −11 points) holds under both protocols.
2. **egoOmni:** the colleague checkpoints lead by a wide margin (`ckpt_sft` 52.3, `ckpt_fft_epoch2` 48.4, `ckpt_epoch2` 45.6 vs base
   36.8; Gemini 3.8 Flash 56.3); our EgoAVU LoRAs (28.7–30.1) fall below the base.
3. **EgoAVU-Bench:** our LoRAs lead (AVSN METEOR 19–21 vs 10.5 for the base and 4–6 for the colleague checkpoints, whose answers are
   much shorter; AVH 97–98).
4. **EgoTaskQA:** the base is best overall; the colleague checkpoints sit between the base and our LoRAs (`ckpt_epoch2`'s indirect judge
   score, 38.65, edges the base's 38.17).
5. **EgoCross:** no model is significantly better than the base; `ckpt_fft_epoch2` is significantly worse (−4.2, p = 0.004).

### Details per benchmark

**EgoAVU-Bench** (S = judge 1–5, M / R = METEOR / ROUGE-L; details [`egoavu_bench/results/`](egoavu_bench/results/README.md))

| model | protocol | SSA S | AVDN S | AVDN M | AVDN R | AVSN S | AVSN M | AVSN R | TR Acc | AVH Acc |
|---|---|---|---|---|---|---|---|---|---|---|
| base | original | 1.50 | 1.78 | 15.93 | 15.53 | 1.92 | 10.47 | 15.30 | 44.6 | 23.6 |
| ckpt_sft | original | 1.53 | 1.73 | 7.99 | 11.92 | 1.68 | 4.39 | 9.75 | 43.6 | 13.0 |
| ckpt_epoch2 | original | 1.56 | 1.70 | 4.93 | 9.91 | 1.67 | 5.11 | 10.76 | 45.0 | 36.1 |
| ckpt_fft_epoch2 | original | 1.56 | 1.72 | 5.66 | 10.50 | 1.69 | 5.46 | 11.42 | 44.8 | 22.5 |
| **r20k-32gpu** | original | 2.62 | 2.03 | 12.57 | 16.76 | 2.06 | 19.57 | 24.76 | 47.2 | 97.1 |
| **r100k** | original | 2.65 | 2.01 | 11.97 | 16.82 | 2.01 | 20.59 | 26.26 | 44.2 | 98.4 |
| **r20k-8gpu** | original | 2.65 | 2.02 | 12.71 | 16.92 | 2.05 | 19.33 | 24.83 | 42.0 | 97.5 |
| base | kit | 1.45 | 1.78 | 15.05 | 15.75 | 1.82 | 11.13 | 15.00 | 42.4 | 29.3 |
| ckpt_epoch2 | kit | 1.54 | 1.66 | 4.83 | 9.76 | 1.65 | 5.19 | 11.07 | 43.6 | 42.5 |
| ckpt_fft_epoch2 | kit | 1.55 | 1.71 | 5.59 | 10.27 | 1.66 | 5.67 | 11.80 | 44.2 | 28.5 |
| paper: base / paper LoRA | — | 1.50 / 3.15 | 2.37 / 2.60 | 10.69 / 12.20 | 14.74 / 17.19 | 1.99 / 2.45 | 9.99 / 22.53 | 13.39 / 28.34 | 53.2 / 64.3 | 42.7 / 61.7 |

All three EgoAVU LoRAs beat the base on every judged category (paired 95% CIs exclude 0); TR accuracy differences are not significant.
AVH yes/no: the LoRAs' gain is the removal of the base's "Yes" bias (base says "Yes" to 80.9 % of probes, our LoRAs to 3–6 %,
`ckpt_fft_epoch2` to 81.9 %).

**egoOmni** (this repo's benchmark; judge Qwen3-32B; 5,176 gold rows = 3,765 original + 136 restored items; `eval/results/`)

| model | gold overall | gold single-turn | gold multi-turn round | gold all-rounds-correct | self overall | self multi-turn round |
|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | **56.26** | 49.97 | **66.25** | **33.52** | **55.74** | **64.90** |
| ckpt_sft (colleague LoRA) | 52.30 | **54.85** | 48.25 | 15.72 | 50.41 | 43.35 |
| ckpt_fft_epoch2 (colleague full FT) | 48.36 | 45.69 | 52.60 | 19.31 | 46.70 | 48.30 |
| ckpt_epoch2 = ckpt_lora_epoch2 (colleague LoRA) | 45.58 | 43.73 | 48.50 | 13.93 | 43.68 | 43.60 |
| video-SALMONN 2+ 72B | 41.17 | 36.74 | 48.20 | 15.03 | 39.76 | 44.55 |
| Qwen2.5-Omni-7B (untuned base) | 36.80 | 33.69 | 41.75 | 8.55 | 35.01 | 37.10 |
| video-SALMONN 2+ 7B | 34.33 | 31.52 | 38.80 | 8.00 | 33.15 | 35.75 |
| MiniCPM-o 2.6 (8B) | 34.31 | 31.33 | 39.05 | 8.28 | 32.07 | 33.25 |
| EgoAVU r20k-8gpu LoRA (ours) | 30.12 | 28.94 | 32.00 | 5.93 | 28.77 | 28.50 |
| EgoAVU r20k-32gpu LoRA (ours) | 29.56 | 29.35 | 29.90 | 5.10 | 27.94 | 25.70 |
| EgoAVU r100k LoRA (ours) | 28.69 | 28.05 | 29.70 | 5.52 | 26.60 | 24.30 |
| VideoLLaMA2.1-7B-AV | 20.81 | 21.35 | 19.95 | 2.07 | 19.15 | 15.65 |

Final-bench-only scores (`eval/results_final/`, 5,152 rows) differ by ≤ 0.13 pt. All twelve were judged by the same local Qwen3-32B
run on H; VideoLLaMA2.1's earlier P-cluster score (20.65) covered the original items only.

**EgoCross** (closed set, official Codabench scoring; [`egocross/`](egocross/README.md))

| model | overall | Surgery | Industry | XSports | Animal | vs base (McNemar) |
|---|---|---|---|---|---|---|
| base | 45.25 | 43.1 | 46.5 | 44.3 | 48.1 | — |
| r100k | 43.89 | 40.3 | 45.3 | 45.5 | 45.4 | −1.36, p = 0.40 |
| r20k-8gpu | 45.66 | 41.7 | 43.7 | 50.0 | 48.6 | +0.42, p = 0.82 |
| r20k-32gpu | 45.14 | 39.2 | 44.9 | 50.4 | 47.5 | −0.10, p = 1.00 |
| ckpt_sft (colleague) | 44.41 | 42.8 | 44.1 | 45.9 | 45.4 | −0.84, p = 0.54 |
| ckpt_epoch2 (colleague) | 43.57 | 41.7 | 46.1 | 42.7 | 44.3 | −1.67, p = 0.23 |
| ckpt_fft_epoch2 (colleague) | 41.07 | 37.1 | 42.0 | 42.3 | 44.3 | −4.18, p = 0.004 |
| base, kit protocol | 45.66 | 43.5 | 46.5 | 47.1 | 45.9 | — (kit reference) |
| ckpt_epoch2, kit protocol | 45.66 | 43.1 | 49.8 | 45.5 | 44.3 | ±0.00 vs kit base, p = 1.00 |
| ckpt_fft_epoch2, kit protocol | 42.22 | 40.6 | 42.5 | 41.5 | 45.4 | −3.44 vs kit base, p = 0.012 |

**EgoSchema** (public 500-question Subset, lmms-eval prompt; [`egoschema/`](egoschema/README.md))

| model | accuracy | vs base (McNemar) |
|---|---|---|
| base | **65.2** | — |
| r20k-8gpu | 64.0 | −1.2, p = 0.59 |
| r20k-32gpu | 60.4 | −4.8, p = 0.02 |
| r100k | 58.0 | −7.2, p = 0.0006 |
| ckpt_fft_epoch2 (colleague) | 57.6 | −7.6, p < 0.0001 |
| ckpt_sft (colleague) | 56.4 | −8.8, p < 0.0001 |
| ckpt_epoch2 (colleague) | 54.2 | −11.0, p < 0.0001 |
| base, kit protocol (360 frames) | 63.8 | — (kit reference) |
| ckpt_fft_epoch2, kit protocol | 56.8 | −7.0 vs kit base, p = 0.0004 |
| ckpt_epoch2, kit protocol | 54.6 | −9.2 vs kit base, p < 0.0001 |

**EgoTaskQA** (test splits; details and definitions in [`egotaskqa/`](egotaskqa/README.md); bold = best under the original protocol)

| model | direct EM | direct judge | direct yes/no | direct open EM / judge | indirect EM | indirect judge | indirect yes/no | indirect open EM / judge |
|---|---|---|---|---|---|---|---|---|
| base | **29.96** | **34.52** | 62.64 | **15.46 / 22.05** | **34.62** | 38.17 | 59.12 | **15.78** / 22.06 |
| r100k | 22.16 | 29.50 | 53.11 | 8.43 / 19.03 | 27.86 | 33.28 | 51.89 | 9.39 / 18.97 |
| r20k-8gpu | 25.06 | 30.88 | 55.45 | 11.59 / 19.98 | 29.77 | 34.17 | 52.75 | 12.11 / 19.88 |
| r20k-32gpu | 24.50 | 30.45 | 54.78 | 11.08 / 19.65 | 29.35 | 34.12 | 52.53 | 11.53 / 19.97 |
| ckpt_sft | 25.65 | 33.31 | **63.79** | 8.74 / 19.80 | 31.02 | 36.81 | 58.33 | 10.02 / 20.26 |
| ckpt_epoch2 | 24.52 | 33.71 | 61.90 | 7.95 / 21.22 | 31.04 | **38.65** | **59.19** | 9.39 / **22.86** |
| ckpt_fft_epoch2 | 23.10 | 32.62 | 62.53 | 5.62 / 19.36 | 29.07 | 36.41 | 58.40 | 6.52 / 19.51 |
| base, kit protocol | 30.07 | 34.41 | 63.05 | 15.45 / 21.71 | 34.60 | 38.33 | 59.14 | 15.73 / 22.33 |
| ckpt_epoch2, kit protocol | 24.42 | 33.58 | 61.42 | 8.02 / 21.23 | 30.84 | 38.33 | 58.89 | 9.27 / 22.53 |
| ckpt_fft_epoch2, kit protocol | 23.22 | 32.64 | 62.94 | 5.60 / 19.21 | 28.99 | 36.26 | 57.94 | 6.72 / 19.58 |

## egoOmni

The original egoOmni test set (`grooLegend/egoOmni` on Hugging Face) has 3765 items = 3040 single-turn EN + 725 multi-turn ZH,
5040 answer turns, 1905 clips; the final test file adds 117 restored items (see [below](#supplement-restored_v3-items-added-2026-09-22)).

| model | weights | predictions | judge |
|---|---|---|---|
| VideoLLaMA2.1-7B-AV | `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV` (repo branch `audio_visual`) | `eval/preds/videollama2_7b_av/` | ✅ `eval/judgments/`, `eval/results/videollama2_7b_av/` |
| video-SALMONN 2+ 7B | `tsinghua-ee/video-SALMONN2_plus_7B_full` | `eval/preds/salmonn2plus_7b/` | ✅ `eval/results/salmonn2plus_7b/` |
| video-SALMONN 2+ 72B | `tsinghua-ee/video-SALMONN2_plus_72B_full` | `eval/preds/salmonn2plus_72b/` | ✅ `eval/results/salmonn2plus_72b/` |
| MiniCPM-o 2.6 (8B) | `openbmb/MiniCPM-o-2_6` | `eval/preds/minicpmo_2_6_8b/` | ✅ `eval/results/minicpmo_2_6_8b/` |
| Gemini 3.8 Flash | API (`gemini-3.8-flash`, default media resolution) | `eval/preds/gemini_3_8_flash/` | ✅ `eval/results/gemini_3_8_flash/` |
| **EgoAVU r100k LoRA (ours)** — Qwen2.5-Omni-7B + LoRA r8, EgoAVU r100k subset, 5 ep | adapter not in this repo | `eval/preds/egoavu_r100k/` | ✅ `eval/judgments/`, `eval/results/egoavu_r100k/` |
| **EgoAVU r20k-8gpu LoRA (ours)** — same recipe, r20k subset, 8 GPUs | adapter not in this repo | `eval/preds/egoavu_r20k8g/` | ✅ `eval/results/egoavu_r20k8g/` |
| **EgoAVU r20k-32gpu LoRA (ours)** — same recipe, r20k subset, 32 GPUs (global batch 32, lr 2.83e-5) | adapter not in this repo | `eval/preds/egoavu_r20k32g/` | ✅ `eval/results/egoavu_r20k32g/` |
| Qwen2.5-Omni-7B (untuned base), same pipeline as the LoRAs | `Qwen/Qwen2.5-Omni-7B` | `eval/preds/qwen25omni7b_base/` | ✅ `eval/results/qwen25omni7b_base/` |
| `ckpt_sft` (colleague LoRA) | not public | `eval/preds/ckpt_sft/` | ✅ `eval/results/ckpt_sft/` |
| `ckpt_epoch2` = `groo_legend/ckpts` `qwen2.5omni7b/ckpt_lora_epoch2` (colleague LoRA) | ModelScope (modelscope.ai) | `eval/preds/ckpt_epoch2/` | ✅ `eval/results/ckpt_epoch2/` |
| `ckpt_fft_epoch2` = `groo_legend/ckpts` `ckpt_fft_epoch2` (colleague full fine-tune) | ModelScope (modelscope.ai) | `eval/preds/ckpt_fft_epoch2/` | ✅ `eval/results/ckpt_fft_epoch2/` |

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
Re-judged on H with the other eleven models, all 5,176 gold rows: overall **20.81 %** (gold) / 19.15 % (self) · single-turn 21.4 % ·
multi-turn round 20.0 % (gold) / 15.7 % (self) · all-rounds-correct 2.1 % · MCQ 68.4 % · open 20.1 % (the first P-cluster run,
original items only: 20.65 / 18.95). Breakdowns in `eval/results/videollama2_7b_av/gold.md`.

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

Predictions of the Qwen2.5-Omni-7B models on all **3,976 QAs** of EgoAVU-Bench (500 videos, 7 categories), laid out for the
official EgoAVU evaluation scripts, and their official scores (see the results above and
[`egoavu_bench/results/`](egoavu_bench/results/README.md)). Full details: [`egoavu_bench/README.md`](egoavu_bench/README.md).

| file | model | protocol |
|---|---|---|
| `egoavu_bench/predictions/ckpt_sft.json`, `ckpt_epoch2.json` | colleague LoRAs | original (P cluster, v2) |
| `egoavu_bench/predictions/r100k.json`, `r20k8g.json`, `r20k32g.json` | EgoAVU LoRAs (ours) | original (H, v3) |
| `egoavu_bench/predictions/ckpt_fft_epoch2.json` | colleague full fine-tune | original (H, 2026-09-25) |
| `egoavu_bench/predictions/kit_base.json`, `kit_ckpt_epoch2.json`, `kit_ckpt_fft_epoch2.json` | base + colleague checkpoints | EgoToM kit (H, 2026-09-25) |
| `egoavu_bench/results/control/base_qwen25omni7b.json` | untuned base | original (P cluster, v2 control) |

- **Protocol (original):** LLaMA-Factory predict; each QA sees only its own `[start_time, end_time]` window; video 2 fps, ≤64 frames,
  ≤200,704 px; 16 kHz audio interleaved with video (`use_audio_in_video`); greedy, ≤1,024 new tokens; every prediction joined
  back to its bench row and verified (3,976 / 3,976 for both models).
- **Scoring with the official scripts needs two fixes:** `llm_as_judge.py` crashes on every input as published (`str.format`
  on a prompt containing a literal JSON block → escape those two braces as `{{ }}`); `captioning_eval.py`'s default categories
  select nothing — pass the full category names, one narration category per run. Join on `bench_idx`, never on `question`
  alone (346 question strings repeat).
- **Input caveats (from the source data, not the pipeline):** 106 windows are fully black (90 also silent), 35 partly black;
  24 windows are shorter than 5 s; Qwen2.5-Omni keeps only the first 300 s of audio, so 822 longer windows are video-only at
  the tail. Lists in `egoavu_bench/notes/bench_window_notes.json`.
