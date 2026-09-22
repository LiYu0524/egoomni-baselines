# EgoAVU-Bench results — v2 (2026-09-23): two Qwen2.5-Omni-7B LoRAs + base-model control, official-standard scoring

> **v2 adds the control**: the untuned **Qwen2.5-Omni-7B** run through the *identical* pipeline (same windows, media settings,
> prompts, greedy decoding, judge). It is the apples-to-apples reference; the paper's own numbers come from a different bench
> version (their "3K QAs / 900 videos" vs the released 3,976 QAs / 500 videos) and a different inference setup.

## 中文摘要
- **流程可信**：未微调基座跑同一套流程后，SSA S 1.502（论文 1.50）、AVSN S 1.917（1.99）、AVSN METEOR 10.47（9.99）、AVDN ROUGE 15.53（14.74），与论文基座基本吻合；TR 44.6 低于论文 53.2，AVDN S 1.78 低于 2.37（bench 版本不同）。
- **两个 LoRA 相对基座没有变好**：AVSN 判分显著下降（−0.24，两者都是）；AVDN/AVSN 的 METEOR/ROUGE 大幅下降（输出太短）；TR 准确率无显著差异；SSA 基本持平。
- **唯一的提升**：`ckpt_epoch2` 在幻觉探针上明显好于基座（是非题 36.1 vs 23.6；AVH-Sound 判分 +0.43 显著）。`ckpt_sft` 反而更差（是非题 13.0，91 % 答 "Yes"）。
- 结论：这两个 checkpoint（训练量远小于论文配方）目前**没有复现论文的微调收益**，主要副作用是回答变短。

## Metrics (official standard)
### Main (paper Table 3 columns)
| model | SSA S | AVDN S | AVDN M | AVDN R | AVSN S | AVSN M | AVSN R | TR Acc | AVH Acc* |
|---|---|---|---|---|---|---|---|---|---|
| **base_qwen25omni7b** | 1.50 | 1.78 | 15.93 | 15.53 | 1.92 | 10.47 | 15.30 | 44.60 | 23.62 |
| **ckpt_sft** | 1.53 | 1.73 | 7.99 | 11.92 | 1.68 | 4.39 | 9.75 | 43.60 | 12.98 |
| **ckpt_epoch2** | 1.56 | 1.70 | 4.93 | 9.91 | 1.67 | 5.11 | 10.76 | 45.00 | 36.10 |
| paper: Qwen2.5-Omni-7B (base) | 1.50 | 2.37 | 10.69 | 14.74 | 1.99 | 9.99 | 13.39 | 53.20 | 42.69 |
| paper: Ours (LoRA) | 3.15 | 2.60 | 12.20 | 17.19 | 2.45 | 22.53 | 28.34 | 64.31 | 61.69 |

### Judge S, all 3,976 items per category (95% CI)
| model | SSA | AVDN | AVSN | TR | AVH-Action | AVH-Object | AVH-Sound |
|---|---|---|---|---|---|---|---|
| base_qwen25omni7b | 1.502 [1.44, 1.56] | 1.782 [1.71, 1.85] | 1.917 [1.87, 1.97] | 2.311 [2.23, 2.40] | 2.551 [2.33, 2.76] | 2.272 [2.05, 2.50] | 1.893 [1.72, 2.06] |
| ckpt_sft | 1.532 [1.47, 1.59] | 1.726 [1.67, 1.79] | 1.681 [1.64, 1.72] | 2.266 [2.18, 2.35] | 2.051 [1.87, 2.26] | 2.000 [1.80, 2.20] | 1.929 [1.75, 2.10] |
| ckpt_epoch2 | 1.563 [1.50, 1.63] | 1.702 [1.64, 1.76] | 1.669 [1.63, 1.71] | 2.283 [2.19, 2.38] | 2.638 [2.42, 2.87] | 2.353 [2.14, 2.58] | 2.327 [2.12, 2.53] |

### Paired differences (same items; 95% CI)
| pair | SSA | AVDN | AVSN | TR | AVH-Action | AVH-Object | AVH-Sound | TR Acc (pts) |
|---|---|---|---|---|---|---|---|---|
| ckpt_sft - base_qwen25omni7b | +0.030 [-0.04, +0.10] | -0.056 [-0.11, +0.00] | -0.236 [-0.28, -0.19] | -0.045 [-0.13, +0.04] | -0.500 [-0.71, -0.30] | -0.272 [-0.46, -0.09] | +0.036 [-0.07, +0.15] | -1.0 [-5.6, +3.4] |
| ckpt_epoch2 - base_qwen25omni7b | +0.062 [-0.01, +0.13] | -0.080 [-0.14, -0.02] | -0.247 [-0.29, -0.20] | -0.028 [-0.11, +0.05] | +0.087 [-0.12, +0.30] | +0.082 [-0.14, +0.30] | +0.434 [+0.23, +0.64] | +0.4 [-3.4, +4.4] |
| ckpt_epoch2 - ckpt_sft | +0.032 [-0.03, +0.10] | -0.024 [-0.08, +0.03] | -0.012 [-0.05, +0.03] | +0.016 [-0.07, +0.09] | +0.587 [+0.36, +0.83] | +0.353 [+0.14, +0.59] | +0.398 [+0.21, +0.60] | +1.4 [-2.4, +5.0] |

### Closed-ended details
| model | TR-MCQ Acc [95% CI] | AVH yes/no Acc Action / Object / Sound | AVH 'Yes' rate | median output/gold words AVSN / AVDN / SSA / TR |
|---|---|---|---|---|
| base_qwen25omni7b | 44.6 [40.4, 49.0] | 38.8 / 32.1 / 0.0 | 80.9 % | 0.51 / 1.94 / 1.21 / 0.64 |
| ckpt_sft | 43.6 [39.2, 47.8] | 15.3 / 22.6 / 1.0 | 91.4 % | 0.19 / 0.44 / 0.54 / 0.38 |
| ckpt_epoch2 | 45.0 [40.6, 49.6] | 44.9 / 43.4 / 20.0 | 67.8 % | 0.22 / 0.23 / 0.57 / 0.36 |

\* **AVH Acc = the 304 yes/no probes only** (macro over Action / Object / Sound). The paper's AVH covers all items per subtype
(196 / 184 / 196) and therefore also scores the 272 open-ended AVH questions, by a rule it does not state. Calibrating that rule on the
base control (`avh_calibration.json`) gives a best fit of "open item correct iff judge rating ≥ 3" — still 6.7 points mean absolute
error against the paper's base (our base 37.3 vs paper 42.7), so the paper's AVH column cannot be reproduced exactly. Under that rule:
base 37.3, `ckpt_sft` 29.3, `ckpt_epoch2` 42.6 — the same ordering as the yes/no column. On the yes/no subset, always answering "No"
would score 95.4 %.

## What v2 establishes
1. **The pipeline is sound.** On the metrics whose definition is unambiguous, the base control matches the paper's base closely
   (SSA 1.502 vs 1.50; AVSN S 1.917 vs 1.99; AVSN M 10.47 vs 9.99; AVDN R 15.53 vs 14.74). TR (44.6 vs 53.2) and AVDN S (1.78 vs 2.37)
   are lower; the released bench differs from the paper's version and our media settings follow the training recipe
   (2 fps / ≤ 64 frames / ≤ 200,704 px) rather than the paper's (1 fps / ≤ 300 frames / 256×256).
2. **Neither LoRA improves on the base**, except for hallucination behaviour in `ckpt_epoch2`. Paired over the same items:
   - AVSN judge S: `ckpt_sft` **−0.236** [−0.28, −0.19], `ckpt_epoch2` **−0.247** [−0.29, −0.20] — both significantly worse.
   - AVDN S: −0.056 [−0.11, 0.00] and −0.080 [−0.14, −0.02]; SSA: +0.03 / +0.06, not significant; TR accuracy: no significant change.
   - METEOR / ROUGE-L collapse (AVSN M 10.47 → 4.4 / 5.1) because the fine-tuned answers are much shorter: median output/gold word
     ratio drops from 0.51 (base) to 0.19 / 0.22 on AVSN and from 1.94 to 0.44 / 0.23 on AVDN.
   - **AVH is the one gain**: `ckpt_epoch2` yes/no 36.1 vs base 23.6 (AVH-Sound judge S +0.434 [+0.23, +0.64]); `ckpt_sft` gets worse
     (13.0), answering "Yes" to 91.4 % of probes whose gold is "No" in 290 / 304.
3. **The two checkpoints differ significantly only on AVH** (`ckpt_epoch2` better); SSA / AVDN / AVSN / TR differences are within CIs.

Likely reasons the LoRAs do not reproduce the paper's gains: far lighter training (`ckpt_epoch2` = epoch 2 of 10 at lr 2e-5, vs the
official LoRA recipe lr 1e-4 × 5 epochs over ~3 M QAs; `ckpt_sft`'s recipe was not shipped), a training answer style much terser than
the bench's gold answers, and no AVDN-style prompts in the training data we checked.

## How each number is produced
| metric | items | method |
|---|---|---|
| S (SSA, AVDN, AVSN) | 600 / 500 / 1,200 | official `evaluation/llm_as_judge.py` logic: judge **Qwen3-235B-A22B-Instruct-2507** (weights SHA-256-verified against ModelScope), upstream `JUDGE_PROMPT` verbatim with its two literal JSON braces escaped (upstream crashes with `KeyError` otherwise), single user turn through the chat template, greedy (upstream's `temperature=0.0` raises in HF because the model's generation config sets `do_sample=true`), `max_new_tokens=512`, `json.loads` → `int(rating)`, per-category mean. vLLM 0.29.0, TP=8, bf16. 0 / 11,928 unparseable across the three models. |
| M, R (AVDN, AVSN) | 500 / 1,200 | official `evaluation/captioning_eval.py`, **unmodified**, once per category. Its CIDEr is 0 by construction (per-sample IDF) and is not reported, as in the paper. |
| TR Acc | 500 four-option MCQ items | answer extraction per the paper ("regex-based string matching … following Yue et al. 2024"; their module is unreleased). Note the answer styles differ: the base model replies with a **letter** (471 / 500), both LoRAs reproduce an **option's text** (459 / 494). Extraction: exact option text → explicit phrase → letter with `:`/`.`/`)` → contained option text → token-F1 best option (≥ 0.6, margin ≥ 0.1). 0 unparsed for all three models. MMMU's parser used verbatim would give 24.8 / 25.6 / 29.6. |
| AVH Acc | 304 yes/no probes | yes / no from the first word. Paper-style variant over all 576 items in `avh_calibration.json`. |

## Files
| file | content |
|---|---|
| `judge_scores.csv`, `judge_scores_base_control.csv` | official judge output format (file_name, category, score) |
| `judge_items/<model>.jsonl` | per item: bench_idx, category, rating, finish_reason, the judge's raw JSON reply (3 models × 3,976) |
| `caption_*.csv` | official `captioning_eval.py` output (Meteor, Rouge; Cider structurally 0) |
| `closed_ended/<model>_closed_items.jsonl` | per TR-MCQ / AVH yes-no item: gold, extracted prediction, extraction rule, correct |
| `avh_calibration.json` | paper-style AVH under each candidate rule, for all three models |
| `control/base_qwen25omni7b.json` | the base control's 3,976 predictions (same schema as `../predictions/`) |
| `summary.json`, `tables.md` | all tables with bootstrap 95 % CIs (2,000 resamples, seed 0) and paired differences |
Reproduce: `tools/judge_official.py`, `tools/closed_ended_acc.py`, `tools/avh_calibrate.py`, `tools/summarize.py`.
