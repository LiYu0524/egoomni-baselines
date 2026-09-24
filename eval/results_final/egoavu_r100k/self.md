# egoavu_r100k — protocol self

rows: 5152 | status: {'ok': 5152} | mean latency 0.0 s

| metric | value |
|---|---|
| overall acc | 0.2657 |
| single-turn acc (n=3152) | 0.2801 |
| multi-turn round acc (n=2000) | 0.2430 |
| multi-turn all-rounds-correct (n=725) | 0.0414 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.7595 |
| open | 5073 | 0.2580 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.3832 |
| loop | 2000 | 0.2430 |
| node | 2252 | 0.2380 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.1551 |
| AE | 235 | 0.2128 |
| AI | 174 | 0.5517 |
| EP | 670 | 0.2254 |
| ES | 345 | 0.4551 |
| GP | 40 | 0.4750 |
| SA | 334 | 0.3892 |
| WI | 117 | 0.4444 |
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
| GP-NXT | 40 | 0.4750 |
| SA-AFF | 130 | 0.4846 |
| SA-CMP | 32 | 0.5000 |
| SA-PRE | 148 | 0.2838 |
| SA-RCH | 24 | 0.3750 |
| WI-NEE | 47 | 0.4255 |
| WI-TRG | 70 | 0.4571 |
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
| A+V | 366 | 0.2568 |
| V | 1712 | 0.2325 |
| unspecified | 3026 | 0.2852 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2928 | 0.2708 |
| video-only clip | 2224 | 0.2590 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2928 | 0.2708 |
| v | 2224 | 0.2590 |

## lang

| group | n | acc |
|---|---|---|
| en | 3256 | 0.2859 |
| zh | 1896 | 0.2310 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.2621 |
| round2 | 725 | 0.2138 |
| round3 | 543 | 0.2541 |
| round4 | 7 | 0.4286 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.2727 |
| 30-120s | 3393 | 0.2767 |
| <30s | 1153 | 0.2281 |
| >10min | 34 | 0.3235 |
