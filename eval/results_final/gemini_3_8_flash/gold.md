# gemini_3_8_flash — protocol gold

rows: 5152 | status: {'ok': 5132, 'unparsed_mcq': 20} | mean latency 35.6 s

| metric | value |
|---|---|
| overall acc | 0.5627 |
| single-turn acc (n=3152) | 0.4994 |
| multi-turn round acc (n=2000) | 0.6625 |
| multi-turn all-rounds-correct (n=725) | 0.3352 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.7215 |
| open | 5073 | 0.5602 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.5698 |
| loop | 2000 | 0.6625 |
| node | 2252 | 0.4707 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.3348 |
| AE | 235 | 0.4043 |
| AI | 174 | 0.6034 |
| EP | 670 | 0.4701 |
| ES | 345 | 0.6754 |
| GP | 40 | 0.6250 |
| SA | 334 | 0.6527 |
| WI | 117 | 0.6068 |
| WS | 347 | 0.6167 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.3727 |
| AD-ORD | 340 | 0.2735 |
| AE-SEN | 208 | 0.3942 |
| AE-STA | 22 | 0.5455 |
| AI-ALT | 126 | 0.6190 |
| AI-IMM | 1 | 0.0000 |
| AI-TSK | 47 | 0.5745 |
| EP-AVC | 63 | 0.2857 |
| EP-AVT | 7 | 0.4286 |
| EP-GAZ | 327 | 0.4526 |
| EP-ID | 9 | 0.7778 |
| EP-SPA | 26 | 0.5000 |
| EP-TRT | 99 | 0.5051 |
| ES-BOD | 230 | 0.6609 |
| ES-LOC | 104 | 0.6923 |
| ES-MEM | 1 | 1.0000 |
| ES-PER | 10 | 0.8000 |
| GP-NXT | 40 | 0.6250 |
| SA-AFF | 130 | 0.7308 |
| SA-CMP | 32 | 0.5938 |
| SA-PRE | 148 | 0.5878 |
| SA-RCH | 24 | 0.7083 |
| WI-NEE | 47 | 0.6596 |
| WI-TRG | 70 | 0.5714 |
| WS-AGT | 15 | 0.5333 |
| WS-CHG | 57 | 0.5439 |
| WS-MOV | 6 | 0.6667 |
| WS-OBJ | 122 | 0.6557 |
| WS-SOC | 5 | 0.6000 |
| WS-SPA | 138 | 0.6232 |
| WS-TMP | 4 | 0.5000 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.3542 |
| A+V | 366 | 0.4754 |
| V | 1712 | 0.4194 |
| unspecified | 3026 | 0.6576 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2928 | 0.5410 |
| video-only clip | 2224 | 0.5913 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2928 | 0.5410 |
| v | 2224 | 0.5913 |

## lang

| group | n | acc |
|---|---|---|
| en | 3256 | 0.5043 |
| zh | 1896 | 0.6630 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.5766 |
| round2 | 725 | 0.6690 |
| round3 | 543 | 0.7643 |
| round4 | 7 | 1.0000 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.5822 |
| 30-120s | 3393 | 0.5270 |
| <30s | 1153 | 0.6583 |
| >10min | 34 | 0.5588 |
