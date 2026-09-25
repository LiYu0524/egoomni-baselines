## Main table (published cells kept; ours fill the blanks)

| Model | EgoSound | EgoToM (paper set, last30sec) | EgoTempo |
|---|---|---|---|
| EgoGPT-7B | 34.30 (pub.) | **63.88** | **16.80** |
| video-SALMONN2+-7B | 36.00 (pub.) | **61.05** | **23.60** |
| MiniCPM-o 2.6-8B | 40.40 (pub.) | **57.51** | **25.20** |
| Qwen2-VL-7B | **31.72** | **61.19** | 26.10 (pub.) |
| LLaVA-OneVision-Qwen2-7B | **33.60** | **63.31** | 23.30 (pub.) |

## EgoToM, all conditions (accuracy %, n = questions answered)

| Model | split | fullcontext | last30sec | last5sec | unparsed | missing preds |
|---|---|---|---|---|---|---|
| EgoGPT-7B | paper | 59.63 (n=706) | 63.88 (n=706) | 60.62 (n=706) | 2 | 0 |
| EgoGPT-7B | all | 59.19 (n=1039) | 63.91 (n=1039) | 60.15 (n=1039) | 2 | 0 |
| video-SALMONN2+-7B | paper | 53.26 (n=706) | 61.05 (n=706) | 63.03 (n=706) | 1 | 0 |
| video-SALMONN2+-7B | all | 54.19 (n=1039) | 61.02 (n=1039) | 63.14 (n=1039) | 3 | 0 |
| MiniCPM-o 2.6-8B | paper | 56.66 (n=706) | 57.51 (n=706) | 57.37 (n=706) | 0 | 0 |
| MiniCPM-o 2.6-8B | all | 55.92 (n=1039) | 57.75 (n=1039) | 57.56 (n=1039) | 0 | 0 |
| Qwen2-VL-7B | paper | 55.67 (n=706) | 61.19 (n=706) | 65.86 (n=706) | 0 | 0 |
| Qwen2-VL-7B | all | 54.38 (n=1039) | 60.15 (n=1039) | 65.26 (n=1039) | 0 | 0 |
| LLaVA-OneVision-Qwen2-7B | paper | 60.20 (n=706) | 63.31 (n=706) | 66.43 (n=706) | 0 | 0 |
| LLaVA-OneVision-Qwen2-7B | all | 58.42 (n=1039) | 63.43 (n=1039) | 64.97 (n=1039) | 0 | 0 |

## Calibration (our pipeline on published cells)

| Model | Bench | Published | Ours | Judge |
|---|---|---|---|---|
| Qwen2-VL-7B | EgoTempo | 26.10 | 23.00 (alt 19.80) | gemini-2.5-flash-lite (alt gemini-2.5-flash) |
| LLaVA-OneVision-Qwen2-7B | EgoTempo | 23.30 | 24.80 (alt 21.80) | gemini-2.5-flash-lite (alt gemini-2.5-flash) |
