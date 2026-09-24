# salmonn2plus_72b — protocol self

rows: 5152 | status: {'ok': 5152} | mean latency 40.24 s

| metric | value |
|---|---|
| overall acc | 0.3967 |
| single-turn acc (n=3152) | 0.3658 |
| multi-turn round acc (n=2000) | 0.4455 |
| multi-turn all-rounds-correct (n=725) | 0.1476 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.8734 |
| open | 5073 | 0.3893 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.4404 |
| loop | 2000 | 0.4455 |
| node | 2252 | 0.3317 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.2438 |
| AE | 235 | 0.2298 |
| AI | 174 | 0.6092 |
| EP | 670 | 0.2836 |
| ES | 345 | 0.5507 |
| GP | 40 | 0.4500 |
| SA | 334 | 0.4760 |
| WI | 117 | 0.5897 |
| WS | 347 | 0.4323 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.2473 |
| AD-ORD | 340 | 0.2382 |
| AE-SEN | 208 | 0.2212 |
| AE-STA | 22 | 0.3636 |
| AI-ALT | 126 | 0.6587 |
| AI-IMM | 1 | 0.0000 |
| AI-TSK | 47 | 0.4894 |
| EP-AVC | 63 | 0.1746 |
| EP-AVT | 7 | 0.0000 |
| EP-GAZ | 327 | 0.2263 |
| EP-ID | 9 | 0.6667 |
| EP-SPA | 26 | 0.5769 |
| EP-TRT | 99 | 0.3939 |
| ES-BOD | 230 | 0.5739 |
| ES-LOC | 104 | 0.5192 |
| ES-MEM | 1 | 1.0000 |
| ES-PER | 10 | 0.3000 |
| GP-NXT | 40 | 0.4500 |
| SA-AFF | 130 | 0.5692 |
| SA-CMP | 32 | 0.5312 |
| SA-PRE | 148 | 0.3716 |
| SA-RCH | 24 | 0.5417 |
| WI-NEE | 47 | 0.6383 |
| WI-TRG | 70 | 0.5571 |
| WS-AGT | 15 | 0.3333 |
| WS-CHG | 57 | 0.4035 |
| WS-MOV | 6 | 0.5000 |
| WS-OBJ | 122 | 0.3607 |
| WS-SOC | 5 | 0.2000 |
| WS-SPA | 138 | 0.5290 |
| WS-TMP | 4 | 0.2500 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.4167 |
| A+V | 366 | 0.3142 |
| V | 1712 | 0.3032 |
| unspecified | 3026 | 0.4594 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2928 | 0.4020 |
| video-only clip | 2224 | 0.3898 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2928 | 0.4020 |
| v | 2224 | 0.3898 |

## lang

| group | n | acc |
|---|---|---|
| en | 3256 | 0.3682 |
| zh | 1896 | 0.4457 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.4331 |
| round2 | 725 | 0.4179 |
| round3 | 543 | 0.4991 |
| round4 | 7 | 0.4286 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.4091 |
| 30-120s | 3393 | 0.3787 |
| <30s | 1153 | 0.4415 |
| >10min | 34 | 0.4706 |
