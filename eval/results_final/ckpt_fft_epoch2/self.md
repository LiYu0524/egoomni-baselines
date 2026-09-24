# ckpt_fft_epoch2 — protocol self

rows: 5152 | status: {'ok': 5152} | mean latency 0.0 s

| metric | value |
|---|---|
| overall acc | 0.4668 |
| single-turn acc (n=3152) | 0.4565 |
| multi-turn round acc (n=2000) | 0.4830 |
| multi-turn all-rounds-correct (n=725) | 0.1945 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.9620 |
| open | 5073 | 0.4591 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.5381 |
| loop | 2000 | 0.4830 |
| node | 2252 | 0.4218 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.3685 |
| AE | 235 | 0.4723 |
| AI | 174 | 0.7069 |
| EP | 670 | 0.4194 |
| ES | 345 | 0.5681 |
| GP | 40 | 0.7000 |
| SA | 334 | 0.4790 |
| WI | 117 | 0.5726 |
| WS | 347 | 0.4179 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.4018 |
| AD-ORD | 340 | 0.3147 |
| AE-SEN | 208 | 0.4856 |
| AE-STA | 22 | 0.4091 |
| AI-ALT | 126 | 0.7937 |
| AI-IMM | 1 | 0.0000 |
| AI-TSK | 47 | 0.4894 |
| EP-AVC | 63 | 0.3016 |
| EP-AVT | 7 | 0.0000 |
| EP-GAZ | 327 | 0.3945 |
| EP-ID | 9 | 0.6667 |
| EP-SPA | 26 | 0.7692 |
| EP-TRT | 99 | 0.4343 |
| ES-BOD | 230 | 0.6087 |
| ES-LOC | 104 | 0.4808 |
| ES-MEM | 1 | 1.0000 |
| ES-PER | 10 | 0.5000 |
| GP-NXT | 40 | 0.7000 |
| SA-AFF | 130 | 0.5462 |
| SA-CMP | 32 | 0.5000 |
| SA-PRE | 148 | 0.3919 |
| SA-RCH | 24 | 0.6250 |
| WI-NEE | 47 | 0.5745 |
| WI-TRG | 70 | 0.5714 |
| WS-AGT | 15 | 0.2000 |
| WS-CHG | 57 | 0.3333 |
| WS-MOV | 6 | 0.5000 |
| WS-OBJ | 122 | 0.3279 |
| WS-SOC | 5 | 0.4000 |
| WS-SPA | 138 | 0.5652 |
| WS-TMP | 4 | 0.0000 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.5208 |
| A+V | 366 | 0.4699 |
| V | 1712 | 0.4328 |
| unspecified | 3026 | 0.4848 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2928 | 0.4570 |
| video-only clip | 2224 | 0.4798 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2928 | 0.4570 |
| v | 2224 | 0.4798 |

## lang

| group | n | acc |
|---|---|---|
| en | 3256 | 0.4598 |
| zh | 1896 | 0.4789 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.4786 |
| round2 | 725 | 0.4759 |
| round3 | 543 | 0.4991 |
| round4 | 7 | 0.4286 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.4720 |
| 30-120s | 3393 | 0.4530 |
| <30s | 1153 | 0.5030 |
| >10min | 34 | 0.5294 |
