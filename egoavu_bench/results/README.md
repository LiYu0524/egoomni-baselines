# EgoAVU-Bench results — v1 (2026-09-23): two Qwen2.5-Omni-7B LoRAs, official-standard scoring

> **Status.** v1 = `ckpt_sft` and `ckpt_epoch2` scored. A control run of the **untuned Qwen2.5-Omni-7B through the identical
> pipeline** is in progress; v2 will add it and calibrate the AVH column (see *AVH* below).

## 中文摘要
- 两个 LoRA 均按官方标准打分：官方 judge Qwen3-235B-A22B-Instruct-2507（权重已与 ModelScope 逐文件 SHA-256 校验），7,952 条 judge 输出 0 条解析失败；METEOR/ROUGE-L 用官方 `captioning_eval.py` 原样计算；TR/AVH 按论文用正则抽取算准确率。
- 两个模型都**远低于论文 LoRA**，多数指标**低于论文中未微调的 Qwen2.5-Omni-7B**。主要原因：回答过短（AVSN 输出/参考 ≈ 0.2）、`ckpt_sft` 在幻觉探针上 91 % 答 "Yes"（参考答案 290/304 为 "No"）、训练量远小于论文、训练集中没有 AVDN 题型。
- 两个 ckpt 之间只有 AVH 差异显著（epoch2 更好），SSA/AVDN/AVSN/TR 的差异都在置信区间内。
- v2 将加入基座对照，用来判断流程与论文是否一致，并校准 AVH 开放题的计分方式。

## Metrics (official standard)
### Main (paper Table 3 columns)
| model | SSA S | AVDN S | AVDN M | AVDN R | AVSN S | AVSN M | AVSN R | TR Acc | AVH Acc* |
|---|---|---|---|---|---|---|---|---|---|
| **ckpt_sft** | 1.53 | 1.73 | 7.99 | 11.92 | 1.68 | 4.39 | 9.75 | 43.60 | 12.98 |
| **ckpt_epoch2** | 1.56 | 1.70 | 4.93 | 9.91 | 1.67 | 5.11 | 10.76 | 45.00 | 36.10 |
| paper: Qwen2.5-Omni-7B (base) | 1.50 | 2.37 | 10.69 | 14.74 | 1.99 | 9.99 | 13.39 | 53.20 | 42.69 |
| paper: Ours (LoRA) | 3.15 | 2.60 | 12.20 | 17.19 | 2.45 | 22.53 | 28.34 | 64.31 | 61.69 |

### Judge S, all 3,976 items per category (95% CI)
| model | SSA | AVDN | AVSN | TR | AVH-Action | AVH-Object | AVH-Sound |
|---|---|---|---|---|---|---|---|
| ckpt_sft | 1.532 [1.47, 1.59] | 1.726 [1.66, 1.79] | 1.681 [1.64, 1.72] | 2.266 [2.18, 2.36] | 2.051 [1.86, 2.25] | 2.000 [1.81, 2.20] | 1.929 [1.76, 2.09] |
| ckpt_epoch2 | 1.563 [1.50, 1.64] | 1.702 [1.64, 1.76] | 1.669 [1.63, 1.71] | 2.283 [2.19, 2.38] | 2.638 [2.42, 2.86] | 2.353 [2.13, 2.59] | 2.327 [2.12, 2.55] |

### Paired differences (same items; 95% CI)
| pair | SSA | AVDN | AVSN | TR | AVH-Action | AVH-Object | AVH-Sound | TR Acc (pts) |
|---|---|---|---|---|---|---|---|---|
| ckpt_epoch2 - ckpt_sft | +0.032 [-0.03, +0.10] | -0.024 [-0.08, +0.03] | -0.012 [-0.05, +0.03] | +0.016 [-0.06, +0.10] | +0.587 [+0.36, +0.82] | +0.353 [+0.14, +0.59] | +0.398 [+0.20, +0.59] | +1.4 [-2.6, +5.4] |

### Closed-ended details
| model | TR-MCQ Acc [95% CI] | AVH yes/no Acc Action / Object / Sound | AVH 'Yes' rate | median output/gold words AVSN / AVDN / SSA / TR |
|---|---|---|---|---|
| ckpt_sft | 43.6 [39.2, 48.0] | 15.3 / 22.6 / 1.0 | 91.4 % | 0.19 / 0.44 / 0.54 / 0.38 |
| ckpt_epoch2 | 45.0 [40.6, 49.4] | 44.9 / 43.4 / 20.0 | 67.8 % | 0.22 / 0.23 / 0.57 / 0.36 |

\* **AVH Acc in v1 = the 304 yes/no probes only**, macro-averaged over Action / Object / Sound. The paper's AVH Acc is a macro over
**all** items per subtype (its values are exactly k/196, k/184, k/196 — the full released subtype sizes), i.e. it also scores the 272
open-ended "What…" AVH questions as right/wrong, with a rule the paper does not specify. So this column is **not yet comparable** to the
paper's 42.69 / 61.69; v2 calibrates it on the base-model control. On the yes/no subset alone, answering "No" always would score 95.4 %.

## How each number is produced
| metric | items | method |
|---|---|---|
| S (SSA, AVDN, AVSN) | 600 / 500 / 1,200 | official `evaluation/llm_as_judge.py` logic: judge **Qwen3-235B-A22B-Instruct-2507**, upstream `JUDGE_PROMPT` verbatim with its two literal JSON braces escaped (upstream crashes with `KeyError` otherwise), single user turn through the chat template, greedy decoding (upstream's `temperature=0.0` raises in HF because the model's generation config sets `do_sample=true`; greedy is its meaning), `max_new_tokens=512`, `json.loads` → `int(rating)`, per-category mean. Run with vLLM 0.29.0, TP=8, bf16 (`tools/judge_official.py`). 0 / 7,952 unparseable, all outputs ended naturally. |
| M, R (AVDN, AVSN) | 500 / 1,200 | official `evaluation/captioning_eval.py` run **unmodified**, once per category (`--categories` = the full category name). CIDEr from that script is 0 by construction (per-sample IDF) and is not reported, as in the paper. |
| TR Acc | 500 four-option MCQ items (gold "The correct option is X…") | answer extraction per the paper ("regex-based string matching … following Yue et al. 2024"; the paper's module is not released). The models answer MCQs by **reproducing an option's text** (459 / 494 of 500), so extraction is: exact option text → explicit phrase → letter with `:`/`.`/`)` → contained option text → token-F1 best option (≥ 0.6, margin ≥ 0.1). 0 unparsed. MMMU's parser used verbatim would give 24.8 / 25.6 (it randomly guesses on short option texts and reads the article "A person…" as option A). The other 600 TR items are open-ended and are covered only by the supplementary judge S. |
| AVH Acc | 304 yes/no probes (v1) | yes / no from the first word (all 608 outputs start with Yes/No). |

## Reading the results
- **Terse outputs.** Median output/gold length ratio is 0.19–0.22 on AVSN; METEOR is recall-weighted, so M/R fall accordingly. On
  length-matched items the ROUGE-L is at or above the paper's base. `ckpt_epoch2`'s dense narrations are much shorter than `ckpt_sft`'s
  (median ratio 0.23 vs 0.44), which is why its AVDN M/R are lower; neither is "better narration".
- **Yes-bias on hallucination probes.** `ckpt_sft` answers "Yes" to 91.4 % of probes whose gold is "No" in 290 / 304 (Sound: 99 %).
  `ckpt_epoch2` reduces that to 67.8 %.
- **TR.** Gold letters are skewed (C = 187 / 500), so "always C" scores 37.4 %; the models are 6–8 points above it.
- **Training budget.** `ckpt_epoch2` is epoch 2 of 10 at lr 2e-5 (the official LoRA recipe is lr 1e-4 × 5 epochs on ~3 M QAs);
  `ckpt_sft`'s recipe was not shipped. AVDN's prompt style never appears in the training data we checked.
- **Inference settings differ from the paper** (ours = training recipe: 2 fps, ≤ 64 frames, ≤ 200,704 px; paper: 1 fps, ≤ 300 frames,
  256×256), and the released bench (3,976 QAs / 500 videos) is a different version from the paper's "3K QAs / 900 videos". The v2
  base-model control measures how much of the gap this explains.
- **Input quality.** 106 windows are black in the Ego4D source (90 also silent); excluding them moves S by ≤ 0.06 and accuracies
  by ≤ 0.7 points. 56 SSA golds say "no significant audio cues" and both models always get those wrong.

## Files
| file | content |
|---|---|
| `judge_scores.csv` | official judge output format (file_name, category, score) — all 7 categories |
| `judge_items/<model>.jsonl` | per item: bench_idx, category, rating, finish_reason, the judge's raw JSON reply |
| `caption_avsn.csv`, `caption_avdn.csv` | official `captioning_eval.py` output (Meteor, Rouge; Cider is structurally 0) |
| `closed_ended/<model>_closed_items.jsonl` | per TR-MCQ / AVH yes-no item: gold, extracted prediction, extraction rule, correct |
| `summary.json`, `tables.md` | everything above with bootstrap 95 % CIs (2,000 resamples, seed 0) and paired differences |
| `judge_model_sha256.tsv`, `judge_run_log_excerpt.txt` | judge weights provenance (all 129 files match ModelScope) and run settings |
Reproduce: `tools/judge_official.py`, `tools/closed_ended_acc.py`, `tools/summarize.py`.
