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
