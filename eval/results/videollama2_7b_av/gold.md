# videollama2_7b_av — protocol gold

rows: 5176 | status: {'ok': 5176} | mean latency 0.93 s

| metric | value |
|---|---|
| overall acc | 0.2081 |
| single-turn acc (n=3176) | 0.2135 |
| multi-turn round acc (n=2000) | 0.1995 |
| multi-turn all-rounds-correct (n=725) | 0.0207 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.6835 |
| open | 5097 | 0.2007 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.2678 |
| loop | 2000 | 0.1995 |
| node | 2252 | 0.1914 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.1247 |
| AE | 235 | 0.1872 |
| AI | 174 | 0.3908 |
| EP | 670 | 0.1716 |
| ES | 345 | 0.3275 |
| GP | 43 | 0.3721 |
| SA | 334 | 0.2665 |
| WI | 138 | 0.2174 |
| WS | 347 | 0.2651 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.1273 |
| AD-ORD | 340 | 0.1206 |
| AE-SEN | 208 | 0.1635 |
| AE-STA | 22 | 0.4091 |
| AI-ALT | 126 | 0.4365 |
| AI-IMM | 1 | 0.0000 |
| AI-TSK | 47 | 0.2766 |
| EP-AVC | 63 | 0.0794 |
| EP-AVT | 7 | 0.1429 |
| EP-GAZ | 327 | 0.1529 |
| EP-ID | 9 | 0.3333 |
| EP-SPA | 26 | 0.5000 |
| EP-TRT | 99 | 0.1818 |
| ES-BOD | 230 | 0.3391 |
| ES-LOC | 104 | 0.3173 |
| ES-MEM | 1 | 0.0000 |
| ES-PER | 10 | 0.2000 |
| GP-NXT | 43 | 0.3721 |
| SA-AFF | 130 | 0.2769 |
| SA-CMP | 32 | 0.3438 |
| SA-PRE | 148 | 0.2230 |
| SA-RCH | 24 | 0.3750 |
| WI-NEE | 64 | 0.2656 |
| WI-TRG | 74 | 0.1757 |
| WS-AGT | 15 | 0.1333 |
| WS-CHG | 57 | 0.1754 |
| WS-MOV | 6 | 0.3333 |
| WS-OBJ | 122 | 0.2377 |
| WS-SOC | 5 | 0.0000 |
| WS-SPA | 138 | 0.3478 |
| WS-TMP | 4 | 0.2500 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.2917 |
| A+V | 377 | 0.1565 |
| V | 1725 | 0.1803 |
| unspecified | 3026 | 0.2290 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2951 | 0.2040 |
| video-only clip | 2225 | 0.2135 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2951 | 0.2040 |
| v | 2225 | 0.2135 |

## lang

| group | n | acc |
|---|---|---|
| en | 3280 | 0.2180 |
| zh | 1896 | 0.1909 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.1366 |
| round2 | 725 | 0.2124 |
| round3 | 543 | 0.2652 |
| round4 | 7 | 0.2857 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.2430 |
| 30-120s | 3417 | 0.2098 |
| <30s | 1153 | 0.1865 |
| >10min | 34 | 0.1765 |
