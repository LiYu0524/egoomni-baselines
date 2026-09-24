# egoavu_r20k8g — protocol gold

rows: 5152 | status: {'ok': 5151, 'unjudged': 1} | mean latency 0.0 s

| metric | value |
|---|---|
| overall acc | 0.3007 |
| single-turn acc (n=3152) | 0.2884 |
| multi-turn round acc (n=2000) | 0.3200 |
| multi-turn all-rounds-correct (n=725) | 0.0593 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.7468 |
| open | 5073 | 0.2937 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.3921 |
| loop | 2000 | 0.3200 |
| node | 2252 | 0.2420 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.1629 |
| AE | 235 | 0.2511 |
| AI | 174 | 0.5345 |
| EP | 670 | 0.2075 |
| ES | 345 | 0.4783 |
| GP | 40 | 0.5500 |
| SA | 334 | 0.3952 |
| WI | 117 | 0.4957 |
| WS | 347 | 0.2767 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.1873 |
| AD-ORD | 340 | 0.1235 |
| AE-SEN | 208 | 0.2356 |
| AE-STA | 22 | 0.4545 |
| AI-ALT | 126 | 0.5952 |
| AI-IMM | 1 | 1.0000 |
| AI-TSK | 47 | 0.3617 |
| EP-AVC | 63 | 0.1905 |
| EP-AVT | 7 | 0.0000 |
| EP-GAZ | 327 | 0.1682 |
| EP-ID | 9 | 0.6667 |
| EP-SPA | 26 | 0.3462 |
| EP-TRT | 99 | 0.2525 |
| ES-BOD | 230 | 0.5000 |
| ES-LOC | 104 | 0.4615 |
| ES-MEM | 1 | 0.0000 |
| ES-PER | 10 | 0.2000 |
| GP-NXT | 40 | 0.5500 |
| SA-AFF | 130 | 0.4615 |
| SA-CMP | 32 | 0.4375 |
| SA-PRE | 148 | 0.3176 |
| SA-RCH | 24 | 0.4583 |
| WI-NEE | 47 | 0.5106 |
| WI-TRG | 70 | 0.4857 |
| WS-AGT | 15 | 0.1333 |
| WS-CHG | 57 | 0.2982 |
| WS-MOV | 6 | 0.5000 |
| WS-OBJ | 122 | 0.2705 |
| WS-SOC | 5 | 0.2000 |
| WS-SPA | 138 | 0.2826 |
| WS-TMP | 4 | 0.2500 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.2500 |
| A+V | 366 | 0.2268 |
| V | 1712 | 0.2459 |
| unspecified | 3026 | 0.3414 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2928 | 0.2917 |
| video-only clip | 2224 | 0.3125 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2928 | 0.2917 |
| v | 2224 | 0.3125 |

## lang

| group | n | acc |
|---|---|---|
| en | 3256 | 0.2942 |
| zh | 1896 | 0.3117 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.2745 |
| round2 | 725 | 0.2966 |
| round3 | 543 | 0.4070 |
| round4 | 7 | 0.7143 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.3199 |
| 30-120s | 3393 | 0.2938 |
| <30s | 1153 | 0.3114 |
| >10min | 34 | 0.2941 |
