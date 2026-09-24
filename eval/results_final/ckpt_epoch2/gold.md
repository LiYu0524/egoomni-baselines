# ckpt_epoch2 — protocol gold

rows: 5152 | status: {'ok': 5152} | mean latency 0.0 s

| metric | value |
|---|---|
| overall acc | 0.4559 |
| single-turn acc (n=3152) | 0.4375 |
| multi-turn round acc (n=2000) | 0.4850 |
| multi-turn all-rounds-correct (n=725) | 0.1393 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.9114 |
| open | 5073 | 0.4488 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.5000 |
| loop | 2000 | 0.4850 |
| node | 2252 | 0.4139 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.3551 |
| AE | 235 | 0.4298 |
| AI | 174 | 0.6437 |
| EP | 670 | 0.4149 |
| ES | 345 | 0.5420 |
| GP | 40 | 0.5250 |
| SA | 334 | 0.4671 |
| WI | 117 | 0.4872 |
| WS | 347 | 0.4352 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.3927 |
| AD-ORD | 340 | 0.2941 |
| AE-SEN | 208 | 0.4375 |
| AE-STA | 22 | 0.4545 |
| AI-ALT | 126 | 0.7222 |
| AI-IMM | 1 | 0.0000 |
| AI-TSK | 47 | 0.4468 |
| EP-AVC | 63 | 0.2540 |
| EP-AVT | 7 | 0.4286 |
| EP-GAZ | 327 | 0.4067 |
| EP-ID | 9 | 0.7778 |
| EP-SPA | 26 | 0.6923 |
| EP-TRT | 99 | 0.4242 |
| ES-BOD | 230 | 0.5826 |
| ES-LOC | 104 | 0.4808 |
| ES-MEM | 1 | 1.0000 |
| ES-PER | 10 | 0.2000 |
| GP-NXT | 40 | 0.5250 |
| SA-AFF | 130 | 0.5538 |
| SA-CMP | 32 | 0.4375 |
| SA-PRE | 148 | 0.3784 |
| SA-RCH | 24 | 0.5833 |
| WI-NEE | 47 | 0.4681 |
| WI-TRG | 70 | 0.5000 |
| WS-AGT | 15 | 0.4000 |
| WS-CHG | 57 | 0.4386 |
| WS-MOV | 6 | 0.5000 |
| WS-OBJ | 122 | 0.3934 |
| WS-SOC | 5 | 0.4000 |
| WS-SPA | 138 | 0.4855 |
| WS-TMP | 4 | 0.0000 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.4792 |
| A+V | 366 | 0.4180 |
| V | 1712 | 0.4141 |
| unspecified | 3026 | 0.4838 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2928 | 0.4532 |
| video-only clip | 2224 | 0.4595 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2928 | 0.4532 |
| v | 2224 | 0.4595 |

## lang

| group | n | acc |
|---|---|---|
| en | 3256 | 0.4410 |
| zh | 1896 | 0.4815 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.4193 |
| round2 | 725 | 0.4497 |
| round3 | 543 | 0.6133 |
| round4 | 7 | 1.0000 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.4720 |
| 30-120s | 3393 | 0.4371 |
| <30s | 1153 | 0.4987 |
| >10min | 34 | 0.6176 |
