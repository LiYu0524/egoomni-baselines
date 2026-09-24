# egoavu_r100k — protocol gold

rows: 5176 | status: {'ok': 5176} | mean latency 0.0 s

| metric | value |
|---|---|
| overall acc | 0.2869 |
| single-turn acc (n=3176) | 0.2805 |
| multi-turn round acc (n=2000) | 0.2970 |
| multi-turn all-rounds-correct (n=725) | 0.0552 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.7595 |
| open | 5097 | 0.2796 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.3832 |
| loop | 2000 | 0.2970 |
| node | 2252 | 0.2380 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.1551 |
| AE | 235 | 0.2128 |
| AI | 174 | 0.5517 |
| EP | 670 | 0.2254 |
| ES | 345 | 0.4551 |
| GP | 43 | 0.4651 |
| SA | 334 | 0.3892 |
| WI | 138 | 0.4275 |
| WS | 347 | 0.2594 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.1782 |
| AD-ORD | 340 | 0.1176 |
| AE-SEN | 208 | 0.2067 |
| AE-STA | 22 | 0.3182 |
| AI-ALT | 126 | 0.6032 |
| AI-IMM | 1 | 1.0000 |
| AI-TSK | 47 | 0.4043 |
| EP-AVC | 63 | 0.2063 |
| EP-AVT | 7 | 0.0000 |
| EP-GAZ | 327 | 0.1590 |
| EP-ID | 9 | 0.7778 |
| EP-SPA | 26 | 0.4231 |
| EP-TRT | 99 | 0.2222 |
| ES-BOD | 230 | 0.4826 |
| ES-LOC | 104 | 0.4327 |
| ES-MEM | 1 | 0.0000 |
| ES-PER | 10 | 0.1000 |
| GP-NXT | 43 | 0.4651 |
| SA-AFF | 130 | 0.4846 |
| SA-CMP | 32 | 0.5000 |
| SA-PRE | 148 | 0.2838 |
| SA-RCH | 24 | 0.3750 |
| WI-NEE | 64 | 0.3906 |
| WI-TRG | 74 | 0.4595 |
| WS-AGT | 15 | 0.2000 |
| WS-CHG | 57 | 0.2456 |
| WS-MOV | 6 | 0.3333 |
| WS-OBJ | 122 | 0.2951 |
| WS-SOC | 5 | 0.0000 |
| WS-SPA | 138 | 0.2464 |
| WS-TMP | 4 | 0.2500 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.2917 |
| A+V | 377 | 0.2573 |
| V | 1725 | 0.2336 |
| unspecified | 3026 | 0.3209 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2951 | 0.2830 |
| video-only clip | 2225 | 0.2921 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2951 | 0.2830 |
| v | 2225 | 0.2921 |

## lang

| group | n | acc |
|---|---|---|
| en | 3280 | 0.2878 |
| zh | 1896 | 0.2853 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.2621 |
| round2 | 725 | 0.2676 |
| round3 | 543 | 0.3812 |
| round4 | 7 | 0.4286 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.2815 |
| 30-120s | 3417 | 0.2903 |
| <30s | 1153 | 0.2793 |
| >10min | 34 | 0.2941 |
