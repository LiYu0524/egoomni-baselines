# qwen25omni7b_base — protocol gold

rows: 5152 | status: {'ok': 5152} | mean latency 0.0 s

| metric | value |
|---|---|
| overall acc | 0.3668 |
| single-turn acc (n=3152) | 0.3347 |
| multi-turn round acc (n=2000) | 0.4175 |
| multi-turn all-rounds-correct (n=725) | 0.0855 |

## format

| group | n | acc |
|---|---|---|
| mcq | 79 | 0.7215 |
| open | 5073 | 0.3613 |

## track

| group | n | acc |
|---|---|---|
| edge | 788 | 0.4340 |
| loop | 2000 | 0.4175 |
| node | 2252 | 0.2926 |

## category

| group | n | acc |
|---|---|---|
| AD | 890 | 0.2067 |
| AE | 235 | 0.3319 |
| AI | 174 | 0.5230 |
| EP | 670 | 0.2701 |
| ES | 345 | 0.4754 |
| GP | 40 | 0.3750 |
| SA | 334 | 0.4641 |
| WI | 117 | 0.4872 |
| WS | 347 | 0.3746 |

## subcategory

| group | n | acc |
|---|---|---|
| AD-ID | 550 | 0.2218 |
| AD-ORD | 340 | 0.1824 |
| AE-SEN | 208 | 0.3269 |
| AE-STA | 22 | 0.4545 |
| AI-ALT | 126 | 0.5794 |
| AI-IMM | 1 | 1.0000 |
| AI-TSK | 47 | 0.3617 |
| EP-AVC | 63 | 0.2063 |
| EP-AVT | 7 | 0.0000 |
| EP-GAZ | 327 | 0.2416 |
| EP-ID | 9 | 0.6667 |
| EP-SPA | 26 | 0.2692 |
| EP-TRT | 99 | 0.3131 |
| ES-BOD | 230 | 0.5261 |
| ES-LOC | 104 | 0.3846 |
| ES-MEM | 1 | 0.0000 |
| ES-PER | 10 | 0.3000 |
| GP-NXT | 40 | 0.3750 |
| SA-AFF | 130 | 0.5077 |
| SA-CMP | 32 | 0.5938 |
| SA-PRE | 148 | 0.3716 |
| SA-RCH | 24 | 0.6250 |
| WI-NEE | 47 | 0.4894 |
| WI-TRG | 70 | 0.4857 |
| WS-AGT | 15 | 0.2667 |
| WS-CHG | 57 | 0.2632 |
| WS-MOV | 6 | 0.5000 |
| WS-OBJ | 122 | 0.3852 |
| WS-SOC | 5 | 0.2000 |
| WS-SPA | 138 | 0.4203 |
| WS-TMP | 4 | 0.5000 |

## min_modalities

| group | n | acc |
|---|---|---|
| A | 48 | 0.2708 |
| A+V | 366 | 0.3087 |
| V | 1712 | 0.2804 |
| unspecified | 3026 | 0.4243 |

## clip_audio

| group | n | acc |
|---|---|---|
| audio+video | 2928 | 0.3610 |
| video-only clip | 2224 | 0.3746 |

## modality_used

| group | n | acc |
|---|---|---|
| av | 2928 | 0.3610 |
| v | 2224 | 0.3746 |

## lang

| group | n | acc |
|---|---|---|
| en | 3256 | 0.3406 |
| zh | 1896 | 0.4119 |

## multi_turn_round

| group | n | acc |
|---|---|---|
| round1 | 725 | 0.3586 |
| round2 | 725 | 0.3890 |
| round3 | 543 | 0.5304 |
| round4 | 7 | 0.7143 |

## clip_duration

| group | n | acc |
|---|---|---|
| 2-10min | 572 | 0.3374 |
| 30-120s | 3393 | 0.3551 |
| <30s | 1153 | 0.4128 |
| >10min | 34 | 0.4706 |
