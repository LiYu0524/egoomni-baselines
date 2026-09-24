# minicpmo_2_6_8b — protocol self

rows: 5176 | status: {'ok': 5176} | mean latency 3.33 s

| metric | value |
|---|---|
| overall acc | 0.3207 |
| single-turn acc (n=3176) | 0.3133 |
| multi-turn round acc (n=2000) | 0.3325 |
| multi-turn all-rounds-correct (n=725) | 0.0828 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.7848 |
| open | 5097 | 0.3135 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.3782 |
| loop | 2000 | 0.3325 |
| node | 2252 | 0.2904 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.2146 |
| AE | 235 | 0.2255 |
| AI | 174 | 0.5345 |
| EP | 670 | 0.2507 |
| ES | 345 | 0.4986 |
| GP | 43 | 0.3488 |
| SA | 334 | 0.3982 |
| WI | 138 | 0.3406 |
| WS | 347 | 0.3545 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.2291 |
| AD-ORD | 340 | 0.1912 |
| AE-SEN | 208 | 0.2115 |
| AE-STA | 22 | 0.4091 |
| AI-ALT | 126 | 0.5794 |
| AI-IMM | 1 | 0.0000 |
| AI-TSK | 47 | 0.4255 |
| EP-AVC | 63 | 0.1270 |
| EP-AVT | 7 | 0.0000 |
| EP-GAZ | 327 | 0.2018 |
| EP-ID | 9 | 0.6667 |
| EP-SPA | 26 | 0.5385 |
| EP-TRT | 99 | 0.3030 |
| ES-BOD | 230 | 0.5304 |
| ES-LOC | 104 | 0.4615 |
| ES-MEM | 1 | 0.0000 |
| ES-PER | 10 | 0.2000 |
| GP-NXT | 43 | 0.3488 |
| SA-AFF | 130 | 0.4846 |
| SA-CMP | 32 | 0.3750 |
| SA-PRE | 148 | 0.2973 |
| SA-RCH | 24 | 0.5833 |
| WI-NEE | 64 | 0.3750 |
| WI-TRG | 74 | 0.3108 |
| WS-AGT | 15 | 0.2000 |
| WS-CHG | 57 | 0.2456 |
| WS-MOV | 6 | 0.6667 |
| WS-OBJ | 122 | 0.3443 |
| WS-SOC | 5 | 0.0000 |
| WS-SPA | 138 | 0.4203 |
| WS-TMP | 4 | 0.5000 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.3333 |
| A+V | 377 | 0.2440 |
| V | 1725 | 0.2661 |
| unspecified | 3026 | 0.3612 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2951 | 0.3182 |
| video-only clip | 2225 | 0.3240 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2951 | 0.3182 |
| v | 2225 | 0.3240 |

## lang

| group | n | acc |
|---|---|---|
| en | 3280 | 0.3162 |
| zh | 1896 | 0.3286 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.3297 |
| round2 | 725 | 0.3366 |
| round3 | 543 | 0.3278 |
| round4 | 7 | 0.5714 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.3444 |
| 30-120s | 3417 | 0.3175 |
| <30s | 1153 | 0.3157 |
| >10min | 34 | 0.4118 |
