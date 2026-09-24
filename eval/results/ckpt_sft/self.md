# ckpt_sft — protocol self

rows: 5176 | status: {'ok': 5176} | mean latency 0.0 s

| metric | value |
|---|---|
| overall acc | 0.5041 |
| single-turn acc (n=3176) | 0.5485 |
| multi-turn round acc (n=2000) | 0.4335 |
| multi-turn all-rounds-correct (n=725) | 0.1476 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.9367 |
| open | 5097 | 0.4974 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.6612 |
| loop | 2000 | 0.4335 |
| node | 2252 | 0.5142 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.4146 |
| AE | 235 | 0.5574 |
| AI | 174 | 0.7931 |
| EP | 670 | 0.5448 |
| ES | 345 | 0.6348 |
| GP | 43 | 0.5581 |
| SA | 334 | 0.6677 |
| WI | 138 | 0.4928 |
| WS | 347 | 0.5908 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.4109 |
| AD-ORD | 340 | 0.4206 |
| AE-SEN | 208 | 0.5481 |
| AE-STA | 22 | 0.5909 |
| AI-ALT | 126 | 0.8095 |
| AI-IMM | 1 | 1.0000 |
| AI-TSK | 47 | 0.7447 |
| EP-AVC | 63 | 0.3016 |
| EP-AVT | 7 | 0.5714 |
| EP-GAZ | 327 | 0.5566 |
| EP-ID | 9 | 0.8889 |
| EP-SPA | 26 | 0.7692 |
| EP-TRT | 99 | 0.5354 |
| ES-BOD | 230 | 0.6609 |
| ES-LOC | 104 | 0.5962 |
| ES-MEM | 1 | 1.0000 |
| ES-PER | 10 | 0.4000 |
| GP-NXT | 43 | 0.5581 |
| SA-AFF | 130 | 0.7308 |
| SA-CMP | 32 | 0.7500 |
| SA-PRE | 148 | 0.5743 |
| SA-RCH | 24 | 0.7917 |
| WI-NEE | 64 | 0.5000 |
| WI-TRG | 74 | 0.4865 |
| WS-AGT | 15 | 0.4000 |
| WS-CHG | 57 | 0.5789 |
| WS-MOV | 6 | 0.5000 |
| WS-OBJ | 122 | 0.5574 |
| WS-SOC | 5 | 0.6000 |
| WS-SPA | 138 | 0.6449 |
| WS-TMP | 4 | 0.7500 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.6042 |
| A+V | 377 | 0.4721 |
| V | 1725 | 0.5148 |
| unspecified | 3026 | 0.5003 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2951 | 0.5032 |
| video-only clip | 2225 | 0.5052 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2951 | 0.5032 |
| v | 2225 | 0.5052 |

## lang

| group | n | acc |
|---|---|---|
| en | 3280 | 0.5494 |
| zh | 1896 | 0.4256 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.4193 |
| round2 | 725 | 0.4193 |
| round3 | 543 | 0.4696 |
| round4 | 7 | 0.5714 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.5122 |
| 30-120s | 3417 | 0.5262 |
| <30s | 1153 | 0.4328 |
| >10min | 34 | 0.5588 |
