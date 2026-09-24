# salmonn2plus_7b — protocol gold

rows: 5176 | status: {'ok': 5175, 'unjudged': 1} | mean latency 6.52 s

| metric | value |
|---|---|
| overall acc | 0.3433 |
| single-turn acc (n=3176) | 0.3152 |
| multi-turn round acc (n=2000) | 0.3880 |
| multi-turn all-rounds-correct (n=725) | 0.0800 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.7595 |
| open | 5097 | 0.3369 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.3680 |
| loop | 2000 | 0.3880 |
| node | 2252 | 0.2869 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.2191 |
| AE | 235 | 0.1617 |
| AI | 174 | 0.5402 |
| EP | 670 | 0.2463 |
| ES | 345 | 0.4609 |
| GP | 43 | 0.4651 |
| SA | 334 | 0.4281 |
| WI | 138 | 0.4348 |
| WS | 347 | 0.3660 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.2291 |
| AD-ORD | 340 | 0.2029 |
| AE-SEN | 208 | 0.1538 |
| AE-STA | 22 | 0.2727 |
| AI-ALT | 126 | 0.5952 |
| AI-IMM | 1 | 0.0000 |
| AI-TSK | 47 | 0.4043 |
| EP-AVC | 63 | 0.1270 |
| EP-AVT | 7 | 0.1429 |
| EP-GAZ | 327 | 0.2018 |
| EP-ID | 9 | 0.5556 |
| EP-SPA | 26 | 0.3846 |
| EP-TRT | 99 | 0.2727 |
| ES-BOD | 230 | 0.4783 |
| ES-LOC | 104 | 0.4423 |
| ES-MEM | 1 | 1.0000 |
| ES-PER | 10 | 0.2000 |
| GP-NXT | 43 | 0.4651 |
| SA-AFF | 130 | 0.5308 |
| SA-CMP | 32 | 0.4688 |
| SA-PRE | 148 | 0.3041 |
| SA-RCH | 24 | 0.5833 |
| WI-NEE | 64 | 0.4688 |
| WI-TRG | 74 | 0.4054 |
| WS-AGT | 15 | 0.3333 |
| WS-CHG | 57 | 0.3333 |
| WS-MOV | 6 | 0.3333 |
| WS-OBJ | 122 | 0.3689 |
| WS-SOC | 5 | 0.2000 |
| WS-SPA | 138 | 0.3913 |
| WS-TMP | 4 | 0.2500 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.2917 |
| A+V | 377 | 0.2599 |
| V | 1725 | 0.2667 |
| unspecified | 3026 | 0.3982 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2951 | 0.3453 |
| video-only clip | 2225 | 0.3407 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2951 | 0.3453 |
| v | 2225 | 0.3407 |

## lang

| group | n | acc |
|---|---|---|
| en | 3280 | 0.3189 |
| zh | 1896 | 0.3855 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.3421 |
| round2 | 725 | 0.3572 |
| round3 | 543 | 0.4843 |
| round4 | 7 | 0.8571 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.3462 |
| 30-120s | 3417 | 0.3301 |
| <30s | 1153 | 0.3799 |
| >10min | 34 | 0.3824 |
