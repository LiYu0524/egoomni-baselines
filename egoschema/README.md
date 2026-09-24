# EgoSchema (public Subset) — untuned Qwen2.5-Omni-7B, the three EgoAVU LoRAs and the two colleague LoRAs

EgoSchema (lmms-lab/egoschema): 3-minute egocentric clips with five-way MCQs. Only the 500-question **Subset** has public answers;
the full 5,031-question MC set is scored server-side and was not run. **The released videos have no audio track**, so this is a
video-only evaluation.

## Protocol
- Media: each video is streamed out of `videos_chunked_0N.zip` and transcoded to the proxy format the EgoAVU LoRAs were trained on
  (`prep_media.py`: fps 2, ~262k px/frame, H.264 veryfast crf 23, GOP 20); LLaMA-Factory reads `<proxy>.mp4#t=0,<dur>` with the
  training recipe (≤64 frames, ≤200,704 px).
- Prompt: lmms-eval `egoschema` — question, the five `A.`–`E.` options on new lines, then
  "Answer with the option's letter from the given choices directly."; greedy, ≤64 new tokens; `qwen2_omni` template.
- Parse: lmms-eval `extract_characters_regex` (strip answer prefixes, first A–E). A lenient parse (option-text match) is also
  computed because every EgoSchema option starts with "C …" (the camera wearer); all six models answered with letters, so both
  parses agree exactly.

## Results (accuracy %, 500 questions)
| model | accuracy | vs base (exact McNemar) |
|---|---|---|
| base (Qwen2.5-Omni-7B) | **65.2** | — |
| r20k-8gpu | 64.0 | −1.2, p = 0.59 |
| r20k-32gpu | 60.4 | −4.8, p = 0.02 |
| r100k | 58.0 | −7.2, p = 0.0006 |
| ckpt_sft (colleague LoRA) | 56.4 | −8.8, p < 0.0001 |
| ckpt_epoch2 (colleague LoRA) | 54.2 | −11.0, p < 0.0001 |

`results/<tag>/outputs.jsonl` holds every raw answer with both parses; `scores.json` the summary.
