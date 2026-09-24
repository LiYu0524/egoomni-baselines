# ckpt_sft — protocol gold

rows: 5152 | status: {'ok': 5152} | mean latency 0.0 s

| metric | value |
|---|---|
| overall acc | 0.5233 |
| single-turn acc (n=3152) | 0.5492 |
| multi-turn round acc (n=2000) | 0.4825 |
| multi-turn all-rounds-correct (n=725) | 0.1572 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.9367 |
| open | 5073 | 0.5169 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.6612 |
| loop | 2000 | 0.4825 |
| node | 2252 | 0.5142 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.4146 |
| AE | 235 | 0.5574 |
| AI | 174 | 0.7931 |
| EP | 670 | 0.5448 |
| ES | 345 | 0.6348 |
| GP | 40 | 0.5500 |
| SA | 334 | 0.6677 |
| WI | 117 | 0.5043 |
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
| GP-NXT | 40 | 0.5500 |
| SA-AFF | 130 | 0.7308 |
| SA-CMP | 32 | 0.7500 |
| SA-PRE | 148 | 0.5743 |
| SA-RCH | 24 | 0.7917 |
| WI-NEE | 47 | 0.5319 |
| WI-TRG | 70 | 0.4857 |
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
| A+V | 366 | 0.4809 |
| V | 1712 | 0.5134 |
| unspecified | 3026 | 0.5327 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2928 | 0.5212 |
| video-only clip | 2224 | 0.5261 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2928 | 0.5212 |
| v | 2224 | 0.5261 |

## lang

| group | n | acc |
|---|---|---|
| en | 3256 | 0.5519 |
| zh | 1896 | 0.4742 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.4193 |
| round2 | 725 | 0.4538 |
| round3 | 543 | 0.6004 |
| round4 | 7 | 0.8571 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.5245 |
| 30-120s | 3393 | 0.5326 |
| <30s | 1153 | 0.4935 |
| >10min | 34 | 0.5882 |
