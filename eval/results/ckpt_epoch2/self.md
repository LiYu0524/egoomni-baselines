# ckpt_epoch2 — protocol self

rows: 5176 | status: {'ok': 5176} | mean latency 0.0 s

| metric | value |
|---|---|
| overall acc | 0.4368 |
| single-turn acc (n=3176) | 0.4373 |
| multi-turn round acc (n=2000) | 0.4360 |
| multi-turn all-rounds-correct (n=725) | 0.1448 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.9114 |
| open | 5097 | 0.4295 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.5000 |
| loop | 2000 | 0.4360 |
| node | 2252 | 0.4139 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.3551 |
| AE | 235 | 0.4298 |
| AI | 174 | 0.6437 |
| EP | 670 | 0.4149 |
| ES | 345 | 0.5420 |
| GP | 43 | 0.5349 |
| SA | 334 | 0.4671 |
| WI | 138 | 0.4710 |
| WS | 347 | 0.4352 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.3927 |
| AD-ORD | 340 | 0.2941 |
| AE-SEN | 208 | 0.4375 |
| AE-STA | 22 | 0.4545 |
| AI-ALT | 126 | 0.7222 |
| AI-IMM | 1 | 0.0000 |
| AI-TSK | 47 | 0.4468 |
| EP-AVC | 63 | 0.2540 |
| EP-AVT | 7 | 0.4286 |
| EP-GAZ | 327 | 0.4067 |
| EP-ID | 9 | 0.7778 |
| EP-SPA | 26 | 0.6923 |
| EP-TRT | 99 | 0.4242 |
| ES-BOD | 230 | 0.5826 |
| ES-LOC | 104 | 0.4808 |
| ES-MEM | 1 | 1.0000 |
| ES-PER | 10 | 0.2000 |
| GP-NXT | 43 | 0.5349 |
| SA-AFF | 130 | 0.5538 |
| SA-CMP | 32 | 0.4375 |
| SA-PRE | 148 | 0.3784 |
| SA-RCH | 24 | 0.5833 |
| WI-NEE | 64 | 0.4062 |
| WI-TRG | 74 | 0.5270 |
| WS-AGT | 15 | 0.4000 |
| WS-CHG | 57 | 0.4386 |
| WS-MOV | 6 | 0.5000 |
| WS-OBJ | 122 | 0.3934 |
| WS-SOC | 5 | 0.4000 |
| WS-SPA | 138 | 0.4855 |
| WS-TMP | 4 | 0.0000 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.4792 |
| A+V | 377 | 0.4138 |
| V | 1725 | 0.4151 |
| unspecified | 3026 | 0.4514 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2951 | 0.4415 |
| video-only clip | 2225 | 0.4306 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2951 | 0.4415 |
| v | 2225 | 0.4306 |

## lang

| group | n | acc |
|---|---|---|
| en | 3280 | 0.4402 |
| zh | 1896 | 0.4309 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.4193 |
| round2 | 725 | 0.4221 |
| round3 | 543 | 0.4770 |
| round4 | 7 | 0.4286 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.4563 |
| 30-120s | 3417 | 0.4299 |
| <30s | 1153 | 0.4432 |
| >10min | 34 | 0.5882 |
