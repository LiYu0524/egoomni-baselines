# egoavu_r20k32g — protocol gold

rows: 5152 | status: {'ok': 5151, 'unjudged': 1} | mean latency 0.0 s

| metric | value |
|---|---|
| overall acc | 0.2950 |
| single-turn acc (n=3152) | 0.2925 |
| multi-turn round acc (n=2000) | 0.2990 |
| multi-turn all-rounds-correct (n=725) | 0.0510 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.7468 |
| open | 5073 | 0.2880 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.3947 |
| loop | 2000 | 0.2990 |
| node | 2252 | 0.2473 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.1697 |
| AE | 235 | 0.2511 |
| AI | 174 | 0.5230 |
| EP | 670 | 0.2164 |
| ES | 345 | 0.4725 |
| GP | 40 | 0.5000 |
| SA | 334 | 0.4072 |
| WI | 117 | 0.5043 |
| WS | 347 | 0.2824 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.1909 |
| AD-ORD | 340 | 0.1353 |
| AE-SEN | 208 | 0.2356 |
| AE-STA | 22 | 0.4091 |
| AI-ALT | 126 | 0.5794 |
| AI-IMM | 1 | 1.0000 |
| AI-TSK | 47 | 0.3617 |
| EP-AVC | 63 | 0.2381 |
| EP-AVT | 7 | 0.0000 |
| EP-GAZ | 327 | 0.1743 |
| EP-ID | 9 | 0.6667 |
| EP-SPA | 26 | 0.3846 |
| EP-TRT | 99 | 0.2525 |
| ES-BOD | 230 | 0.4957 |
| ES-LOC | 104 | 0.4615 |
| ES-MEM | 1 | 0.0000 |
| ES-PER | 10 | 0.1000 |
| GP-NXT | 40 | 0.5000 |
| SA-AFF | 130 | 0.4692 |
| SA-CMP | 32 | 0.4375 |
| SA-PRE | 148 | 0.3378 |
| SA-RCH | 24 | 0.4583 |
| WI-NEE | 47 | 0.5745 |
| WI-TRG | 70 | 0.4571 |
| WS-AGT | 15 | 0.1333 |
| WS-CHG | 57 | 0.2105 |
| WS-MOV | 6 | 0.5000 |
| WS-OBJ | 122 | 0.3033 |
| WS-SOC | 5 | 0.4000 |
| WS-SPA | 138 | 0.2971 |
| WS-TMP | 4 | 0.2500 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.2500 |
| A+V | 366 | 0.2404 |
| V | 1712 | 0.2482 |
| unspecified | 3026 | 0.3288 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2928 | 0.2923 |
| video-only clip | 2224 | 0.2986 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2928 | 0.2923 |
| v | 2224 | 0.2986 |

## lang

| group | n | acc |
|---|---|---|
| en | 3256 | 0.2976 |
| zh | 1896 | 0.2906 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.2469 |
| round2 | 725 | 0.2662 |
| round3 | 543 | 0.4088 |
| round4 | 7 | 0.5714 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.2902 |
| 30-120s | 3393 | 0.2977 |
| <30s | 1153 | 0.2879 |
| >10min | 34 | 0.3529 |
