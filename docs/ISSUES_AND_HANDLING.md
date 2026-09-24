# Problems encountered and how they were handled

This file records every problem hit while setting up and running the egoOmni baselines (2026-09-18 → 2026-09-22), what was
done about it, and what is still open. Numbers are measured on the run machine (8×A100-80GB) unless stated otherwise.

## 0. Status at a glance

| | status |
|---|---|
| Predictions, original 3765 items (5040 `gold` + 2000 `self` rows) | complete for VideoLLaMA2.1-AV-7B, video-SALMONN 2+ 7B / 72B, MiniCPM-o 2.6, Gemini 3.8 Flash, EgoAVU r100k — 0 unresolved errors |
| Predictions, `restored_v3` supplement (168 items added on 2026-09-22) | **136 items** complete on all six models (five baselines + EgoAVU r100k); **32 items have no results** because their video is not available (§8) |
| Against the **final** test file (3,882 QAs = 3,765 original + 117 restored) | 3,877 / 3,882 predicted on all six models; the 5 missing are restored items without video; 24 evaluated restored items are not in the final set (§8.1) |
| LLM-judge scores | only VideoLLaMA2.1-AV-7B is judged (Qwen3-32B). Judging the others was paused at the owner's request (§7) |
| Not done | the CPU-vs-GPU preprocessing deviation study for SALMONN (§4.3); gpt-6-astra (cost estimate only, §10) |
| Separate benchmark | EgoAVU-Bench outputs of two other LoRAs are in [`egoavu_bench/`](../egoavu_bench/README.md) (documented there) |

---

## 1. Getting weights and data onto the machine

**Problem.** `huggingface.co` is not reachable from the cluster. `hf-mirror.com` answers metadata requests, but large files
redirect to `cas-bridge.xethub.hf.co`, which is also blocked — downloads start and then sit at 0 B/s forever. The dataset
download had been started through a personal proxy tunnel that died with its SSH session and was retrying in a loop.

**Handling.**
- Everything mirrored on ModelScope was downloaded from there directly (~45 MB/s): VideoLLaMA2.1-7B-AV, SigLIP, Qwen2.5-VL
  7B/72B, whisper-large-v3, MiniCPM-o 2.6. File lists were compared against the HF repos before use.
- Repos that exist only on HF (`tsinghua-ee/*` for SALMONN) and the egoOmni dataset went through the institutional HTTP proxy.
- `scripts/download_models.sh` wraps every download in a retry loop and writes a `.DOWNLOAD_COMPLETE` marker, so a dropped
  connection resumes instead of restarting. The dataset is pinned to revision `63464384` (1905 clips, `test/qa.json`).

## 2. Environments

| problem | handling |
|---|---|
| The shared persistent disk is ~85× slower than local NVMe on small files (2.8 ms vs 0.03 ms per file); a torch env has ~50k files | runtime conda envs live on local NVMe; `scripts/persist_envs.sh` / `restore_envs.sh` keep single-file tarballs on the persistent disk |
| VideoLLaMA2 (torch 2.2 / transformers 4.42) and SALMONN 2+ (torch 2.7 / transformers 4.51) cannot share an env | one env per repo (`scripts/build_env_*.sh`), plus separate envs for MiniCPM-o (model-card pins) and the vLLM judge |
| SALMONN pins `torchaudio 2.5.1`, which does not work with its own `torch 2.7.1` | torchaudio 2.7.1 |
| SALMONN pins `flash_attn 2.7.4.post1`, which has no prebuilt wheel for torch 2.7 | flash-attn 2.8.0.post2 (API-compatible) |
| `torchcodec 0.4.0` only loads FFmpeg 4–7; conda installs FFmpeg 8 → `import torchcodec` fails | `ffmpeg=7.*` pinned in the env |
| VideoLLaMA2's `imageio-ffmpeg 0.4.9` imports `pkg_resources` (removed in setuptools ≥ 81) | `setuptools<81` |
| vLLM's compile step shells out to `ninja`, which was not on PATH | `ninja` installed into the judge env, env `bin/` put on PATH |

## 3. VideoLLaMA2.1-AV-7B

- **Config pointed at non-local paths.** `config.json` names the vision tower by HF id and the audio tower by a relative path
  from the authors' training run. Patched `mm_vision_tower` to the local SigLIP copy (original kept as `config.json.orig`);
  the audio tower weights are already inside the checkpoint.
- **Audio silently dropped.** The repo extracts audio by shelling out to `ffmpeg`; when it is not on PATH it prints
  `Failed to process audio from video` and **continues video-only**. The first smoke test hit exactly this. Fix: every adapter
  puts its env's `bin/` (which contains ffmpeg) on PATH, and every prediction row records `modality_used`
  (`av` / `v`) so a silent fallback would be visible.

## 4. video-SALMONN 2+ (7B, 72B)

### 4.1 Which weights to evaluate
The README's evaluation recipe is "audio-aligned base (`*_audioAlign`) + LoRA (`video-SALMONN-2_plus_*`)". Rebuilding every
tensor of the official merged checkpoint `video-SALMONN2_plus_7B_full` (shard 1, 956 tensors) from base + LoRA
(`W + B·A·α/r`, plus `modules_to_save`) matches **bit-for-bit except `audio.q_tokens`** (the Q-Former query; relative error
0.42). `q_tokens` is a bare `nn.Parameter`, which PEFT's `modules_to_save` cannot store, so the adapter never contained it;
building a base with the repo's `gen_audio_model.py` gives random `q_tokens` instead. **Decision: evaluate the official
`_full` checkpoints.** Validation scripts: `scripts/salmonn_validate_recipe.py`, `scripts/salmonn_validate_shard.py`.

### 4.2 Multi-turn prompts
The repo's test path cuts the prompt at `sum(labels == IGNORE_INDEX)`, which is only correct for single-turn samples. The
adapter cuts at the last `<|im_start|>assistant\n` instead (identical for single-turn).

### 4.3 Throughput, and the one numeric deviation
With the repo's pipeline each clip spent 40–80 s in CPU frame resize/normalization while the GPUs sat at 0 % (48.6 s per
generation). Handling: frames are resized/normalized on the GPU (`processor.preprocess(device="cuda")`) in 64-frame chunks
(chunked output verified bit-identical to whole-video preprocessing; the chunks also fix a 42 GB OOM on 768 frames of
2560×1920), decoding runs on prefetch threads, and a decoded clip is reused for all turns of its item → **8.5 s per
generation**. Cost: pixel values differ from the repo's CPU uint8 path by mean 9.6e-4 (max 0.09; the CPU uint8 kernel itself
differs from float math by up to 0.19). On a 16-item check frame sampling and token counts were identical and 12/16 answers
were word-identical, 4 were paraphrases. A dev300 comparison of judged accuracy between the two paths was planned but **not
run** (judging was paused). `--no_gpu_preprocess` restores the exact repo path.

### 4.4 Errors hidden by the repo's dataset class
On any decode failure the repo's dataset silently substitutes a random other sample (`should_use=False`). The adapter calls
the processing functions directly so a failure becomes a recorded error row instead.

### 4.5 72B
bf16 weights are ~145 GB. `device_map="auto"` (pipeline over GPUs) was estimated at ~3 days, so the 72B runs DeepSpeed
ZeRO-3 data-parallel over 8 ranks, the mechanism of the repo's `test_8.sh`. **Deadlock:** ZeRO-3 gathers parameters
collectively, so all ranks must execute the same modules; a rank holding a clip without audio skips the audio encoder and the
others wait forever — the first smoke test produced nothing in 25 min. Handling: items are processed in two lockstep phases
(clips with audio, then without), each padded with dummy generations so every rank makes the same number of calls. A
DeepSpeed prefetch/persistence setting was A/B-tested on a smoke set and the faster one used. Full run: 10.7 h, 0 errors.

## 5. MiniCPM-o 2.6

Run in the model card's omni mode (1-second units of frame + audio). Deviations, all documented in `eval/README.md`: clips
longer than 128 s are uniformly subsampled to 128 units (the context is 32k tokens; a 33-min clip would need ~200k); greedy
decoding (`num_beams=1, repetition_penalty=1.0`) instead of the card's `sampling=False` default (beam 3, penalty 1.2), so all
baselines share one decoding setting; audio extracted with ffmpeg rather than moviepy (same 16 kHz mono PCM). The run was
interrupted once when another training job started on the same GPUs and killed its workers; workers are resumable, so it
continued from the rows already written.

## 6. Gemini 3.8 Flash (API)

| problem | handling |
|---|---|
| The gateway has no Files API | each clip is re-encoded once (2 fps, ≤1280 px, CRF 26, AAC 64k mono; a ladder of lower settings for clips that would exceed ~14 MB) and sent inline as base64; encodes are cached and reused across an item's turns |
| Thinking is on by default; a small output cap truncates before the answer | `maxOutputTokens` 2048 |
| Two workers transcoding the same clip (clips are shared by several items) collided on a temp file, then a re-check recursed | per-clip `flock` around check → encode → write. The 3 rows that failed before the fix were re-run; the failed rows are kept as an audit trail and `score.py` uses the successful row |
| Gateway throttling (13 → 30 s per request) | 12 → 24 workers; resume reads all shard files so the worker count can change mid-run |
| The API itemizes `AUDIO` in `promptTokensDetails` for only ~2/3 of requests and otherwise folds those tokens into `VIDEO` | verified it is reporting only (identical token totals, identical ~88 tokens/s, 40/40 sampled encodes carry audio). `modality_used` is taken from the media sent, not from the usage report; 1168 rows of the first run were corrected by `eval/repair_gemini_modality.py`; `audio_itemized` records the reporting style |
| Budget control | per-row `cost_usd`; `eval/cost_monitor.py` writes `STOP_API` at the cap, which makes every worker exit cleanly. Spend: **$35.14** for 6315 requests (main set) + **$0.52** for `restored_v3` |
| On 2026-09-22 the gateway was ~5× slower (~160 s per request) and two workers each had ~9 items queued behind very slow requests | the remaining items were re-spread over 12 fresh workers (`run_gemini_restored_v3_resume.sh`) |

## 7. LLM judge

Open answers are scored by a local Qwen3-32B (vLLM) with a binary JSON verdict; MCQ by letter match. VideoLLaMA2.1-AV-7B is
fully judged (6218 unique answers, 0 unparseable verdicts; overall 20.65 % gold / 18.95 % self). Judging SALMONN 7B failed
when the judge was started beside the 72B inference (vLLM: `No available memory for the cache blocks`); before it was
rescheduled, the owner asked to finish all inference first, so **the other models have predictions but no scores yet**.
To score them: `CUDA_VISIBLE_DEVICES=0,1,2,3 ./eval/launch_judge.sh <tag>` then `python -m egoomni_eval.score --tag <tag>`.
Identical (key, prediction) pairs are judged once and cached.

## 8. `restored_v3` supplement — including the 32 items without results

**What changed.** The v9 test file has 3933 rows: the original 3765 items and 168 restored items. For the 3765, questions,
answers, options and multi-turn turns are identical to the evaluated `qa.json` (only evidence-interval metadata and number
formatting differ), so all earlier predictions stay valid. The 168 restored items (138 WI, 30 GP; previously `reject` 85 /
`needs_review` 83; `quality_verified: false`) come **without a clip** — only a window `original_qa.video_clip_range` in the
full Ego4D video. The dataset on HF has no new clips.

**Handling.** `eval/prepare_restored.py` cuts each window, in this order of preference:
1. from one existing egoOmni clip of the same video (clip time = source time − `source_clip_interval.start`) — **115 items**;
2. from the full Ego4D video — **20 items**. 13 of the needed source videos were found on another cluster machine and copied.
   Their timelines were checked against existing clips: 11 matched at offset 0; the other 2 matched at no offset within
   ±120 s until it turned out they are 2880-wide side-by-side recordings and the dataset clips use the **left half** — with
   that crop they match at offset 0 as well. Every full-source cut was then compared with an overlapping dataset clip
   (mean pixel difference 0.1–0.5 of 255 at offset 0 versus 20–100 at ±0.5 s);
3. by stitching two overlapping clips — **1 item** (the two clips were exported at different resolutions, 1280×720 and
   1920×1080; both are scaled to the larger);
4. otherwise skipped — **32 items**.

All cuts are H.264 CRF 18 + AAC 128k; clip durations match the windows within 33 ms. Per-item provenance is in
`eval/data/restored_v3/restored_v3_manifest.json`.

**Caveats for the 136 evaluated items.** The file presents each restored item as a standalone single-turn question and it is
evaluated that way, but 56 of the 136 were originally later rounds of a dialogue (`depends_on_earlier_rounds: true`), e.g. the
GP items ask "To complete that goal, what should I do next?" without the earlier turn. `minimum_modalities` for these items
is derived from `original_qa.loop_annotation.required_modalities` (V 84 / A+V 50 / A 2), since the rows do not carry it.

### 8.1 Final test file
The final `test.qa.jsonl` arrived after this run: 3,882 rows = the same 3,765 original items (content-identical to the previous
file) + **117 of the 168** restored items (the 117 are unchanged). Relative to what was run: 112 of the 117 are evaluated on
all six models; **5 are among the 32 unevaluable items and remain without results**; 27 of the 32 unevaluable items were
dropped; and 24 items that were evaluated (20 cut from full Ego4D videos, 3 from single clips, 1 stitched) were dropped too —
their predictions stay in the repo, flagged `in_final_bench: false` in the manifest, and should be excluded when scoring the
final set (`eval/data/restored_v3/final_bench_restored_ids.json` lists the 117 final ids).

### The 32 items without results

**Why:** their window lies (partly or entirely) outside every clip in the egoOmni dataset, and the full Ego4D video is not
available — it is neither on HF nor on the cluster, and the Ego4D download credentials that were tried are no longer valid.
11 of the 32 windows fall entirely outside the existing clips; for the rest the "missing" column is how many seconds of the
window no clip covers. They involve 20 videos. **To finish them:** place the full videos (or the cut clips) under
`ego4d_full/<video_id>.mp4` and rerun `eval/prepare_restored.py` and the inference scripts; everything else is resumable.

| # | item (`sample_id` without prefix) | category | window in source video (s) | missing (s) | in final test set | question |
|---|---|---|---|---|---|---|
| 1 | `0df89feb-74a6-48ca-9fe9-2c81fc1306e2__0028` | WI / WI-TRG | 159.4–191.4 | 1.0 | no (dropped) | Now that the air pump is no longer attached to the tire, what am I trying to accomplish? |
| 2 | `28170c86-29ba-43e8-8699-e76161f16b98__0024` | WI / WI-NEE | 545.8–596.3 | 38.4 | no (dropped) | Immediately after I am moving forward along the street with clear space ahead of traffic, how does that situat… |
| 3 | `2cb81c1d-9472-4d82-8ec1-72ae355c9163__0027` | WI / WI-NEE | 164.0–196.0 | 4.0 | no (dropped) | When I hear an offscreen person providing information immediately before the next writing phase, how does that… |
| 4 | `2cb81c1d-9472-4d82-8ec1-72ae355c9163__0024` | WI / WI-TRG | 998.8–1030.8 | 32.0 | no (dropped) | How does the open notebook page shape what I am trying to accomplish? |
| 5 | `2d29b45a-169b-453f-8050-98ec147b0ccf__0026` | WI / WI-NEE | 1411.0–1443.0 | 32.0 | no (dropped) | Immediately after I receive feedback from the prior cut, how does seeing the wood piece aligned with the blade… |
| 6 | `3c03af71-426c-4f59-8a20-d2f506359e12__0025` | WI / WI-NEE | 41.1–151.9 | 78.8 | no (dropped) | Immediately after I complete a stitch, when the orange thread remains slack above the embroidery hoop, how doe… |
| 7 | `49d20f86-b516-49be-a27b-a450955c9f46__0026` | WI / WI-NEE | 819.8–858.6 | 38.8 | no (dropped) | With a new page spread visible in the open book, what does this condition lead me to intend to do next? |
| 8 | `54707a82-2fe4-47d2-9456-d5635e358b09__0028` | WI / WI-NEE | 119.6–151.6 | 32.0 | no (dropped) | Immediately after the bristle swipe along the trim ends, when my brush is dry or low on paint, how does that c… |
| 9 | `5f77b76b-24d9-4489-9561-861fc66a1917__0030` | WI / WI-TRG | 550.5–582.5 | 32.0 | **yes — no result** | Immediately after I see the bare section of the cardboard tube that still needs twine, how does that visible c… |
| 10 | `7aa17dca-440f-4845-9831-24555a970de5__0026` | WI / WI-NEE | 475.8–521.9 | 29.1 | no (dropped) | Immediately as I approach the road and the zebra-crossing markings, how does that condition shape what I inten… |
| 11 | `7aa17dca-440f-4845-9831-24555a970de5__0025` | WI / WI-NEE | 1458.0–1490.0 | 20.0 | no (dropped) | Immediately after I am traveling straight forward down the empty street at night, how does that continuing rou… |
| 12 | `9827a535-6963-434f-ad79-1940aa622c22__0022` | WI / WI-TRG | 326.8–358.8 | 23.2 | no (dropped) | How does the dog sitting down in front of me shape what I intend to do? |
| 13 | `9827a535-6963-434f-ad79-1940aa622c22__0021` | WI / WI-NEE | 335.8–377.1 | 14.2 | no (dropped) | After my dog and I have successfully stepped onto the opposite sidewalk, how does that new situation shape wha… |
| 14 | `9827a535-6963-434f-ad79-1940aa622c22__0023` | WI / WI-NEE | 813.3–849.0 | 10.8 | no (dropped) | With a clear sidewalk ahead, what am I trying to accomplish? |
| 15 | `a1d36201-51e6-44db-9d3c-2bff8b1ff268__0024` | WI / WI-NEE | 229.7–286.9 | 39.1 | no (dropped) | How does seeing the paint tray and roller loaded with paint inform what I intend to do next? |
| 16 | `a1d36201-51e6-44db-9d3c-2bff8b1ff268__0027` | WI / WI-NEE | 582.9–614.9 | 3.6 | no (dropped) | When I see that the roller is loaded with fresh paint, how does that condition shape what I am trying to accom… |
| 17 | `ad08b4d6-3f23-4f3c-974c-5d9d90a727b6__0025` | WI / WI-NEE | 1092.1–1149.7 | 57.6 | no (dropped) | Because the dog is maintaining a steady pace ahead along the street, what is my current intention? |
| 18 | `b75bf090-1439-4b11-862b-d1dadab7f854__0024` | WI / WI-TRG | 1110.0–1142.0 | 32.0 | no (dropped) | How did finding that the beverage can was empty shape what I intended to do with it? |
| 19 | `c398a6bf-58cf-4318-9b98-babf649827e5__0028` | WI / WI-TRG | 41.4–73.4 | 0.8 | **yes — no result** | Immediately after noticing that the extension pole's collar had been released or adjusted, how did that condit… |
| 20 | `c7d5d40f-840c-4be0-b79d-ab41394479a2__0030` | WI / WI-NEE | 326.6–358.6 | 14.7 | no (dropped) | Right after I see another player add a playing card to the table, how does that changed game situation shape w… |
| 21 | `c7d5d40f-840c-4be0-b79d-ab41394479a2__0028` | WI / WI-TRG | 719.9–751.9 | 32.0 | no (dropped) | How does the visible arrangement of the playing cards shape what I am trying to accomplish? |
| 22 | `d70be6cb-867f-4657-9a87-5d44ab30527a__0026` | WI / WI-NEE | 864.9–935.4 | 51.1 | no (dropped) | Immediately after the preceding rope adjustment, how does the need to check or guide the rope's position shape… |
| 23 | `eab3bfc0-d611-4a40-89b4-7f9747b55e6a__0027` | WI / WI-TRG | 693.5–725.5 | 1.9 | **yes — no result** | Immediately before I act, when the notebook is resting slanted across the textbook, how does that visible cond… |
| 24 | `eab3bfc0-d611-4a40-89b4-7f9747b55e6a__0028` | WI / WI-NEE | 693.5–725.5 | 1.9 | **yes — no result** | With the textbook open beside my notebook, what do I intend to do next? |
| 25 | `eab3bfc0-d611-4a40-89b4-7f9747b55e6a__0026` | WI / WI-TRG | 895.9–927.9 | 32.0 | **yes — no result** | How does the textbook alongside the open notebook on the desk lead to my intention? |
| 26 | `f2efd904-9a3b-4a59-b1fc-dfa982de920c__0024` | GP / GP-NXT | 0.0–62.9 | 30.9 | no (dropped) | With the phone recording, what broad action should I take next to begin my activity? |
| 27 | `f2efd904-9a3b-4a59-b1fc-dfa982de920c__0025` | WI / WI-TRG | 213.6–267.6 | 19.9 | no (dropped) | How did the wooden platform directly in front of me influence my intention? |
| 28 | `fbbfb7ed-4414-42b1-a22e-35e1a1c51647__0025` | WI / WI-TRG | 626.0–658.0 | 32.0 | no (dropped) | How did seeing the computer case's internal components from the main chamber side shape what I intended to ins… |
| 29 | `fc23c55f-0070-4e96-b970-7d42cb591d02__0025` | WI / WI-NEE | 335.8–367.8 | 25.3 | no (dropped) | After another person asks, "Which one?", what do I intend to do? |
| 30 | `fc23c55f-0070-4e96-b970-7d42cb591d02__0027` | WI / WI-TRG | 335.8–367.8 | 25.3 | no (dropped) | When someone asks, “Which one?”, what does that question prompt me to intend? |
| 31 | `fc23c55f-0070-4e96-b970-7d42cb591d02__0024` | GP / GP-NXT | 711.4–850.9 | 123.9 | no (dropped) | To continue the game, what broad action should I take next? |
| 32 | `fc23c55f-0070-4e96-b970-7d42cb591d02__0026` | WI / WI-TRG | 1434.7–1466.7 | 32.0 | no (dropped) | Immediately before I act, how does seeing a playing card on the table without a game token shape what I intend… |

## 9. EgoAVU r100k LoRA (our model)

Run by a second agent session in parallel with §8; the notes below are from its pipeline (`eval/r100k/`) and were checked
here (7176 rows = the expected (key, protocol) set; missing / extra / duplicate / error / empty rows all 0).

- **Model.** Qwen2.5-Omni-7B thinker (talker / audio output disabled) + the final LoRA of the EgoAVU r100k run
  (`checkpoint-6035`, epoch 5). The adapter weights are not in this repo.
- **Why LLaMAFactory instead of a harness adapter.** The LoRA was trained with LLaMAFactory, so inference uses LLaMAFactory's
  predict mode with the training media settings (2 fps, ≤64 frames, ≤200,704 px per frame, `use_audio_in_video`, `qwen2_omni`
  template, cutoff 32768) and the harness's prompts and decoding (greedy, ≤256 new tokens). `build_data.py` writes the items in
  the training format, `collect.py` converts the outputs back to the harness row schema.
- **Frame counting.** Videos are passed as `<clip>.mp4#t=0,<duration>`. The plain LLaMAFactory path trusts the container's
  `frames` / `duration` fields; for the dataset's VP9 clips these can be 0/None, which would make it treat the clip as endless
  and sample only the first 64 frames. The `#t=` window path counts frames from packet timestamps, which is also the code path
  used in training.
- **Audio.** LLaMAFactory loads audio with torchaudio, which does not read the AAC-in-mp4 tracks here, so audio was
  pre-extracted to 16 kHz mono FLAC (the training format) for the 1213 clips that have audio (0 errors). Clips without audio get
  a `<video>`-only prompt, as in training.
- **Multi-turn `self` protocol.** Shards are item-level (all turns of an item in one job), so each job runs `gold`, then `self`
  rounds 2 → 3 → 4, each built from that job's own earlier answers; `self` round 1 is a copy of `gold` round 1 (the harness
  convention).
- **Join safety.** LLaMAFactory's `generated_predictions.jsonl` is matched to items only by order, so `collect.py` asserts for
  every row that the label equals the gold answer of that turn and that the question appears in the prompt (all 7176 passed).
- **Race fixed before submission.** Parallel jobs originally rewrote the shared `dataset_info.json` (read-modify-write race,
  plus a missing-directory bug); all 16 dataset splits are now registered once up front.
- **Throughput.** 4 cluster jobs × 8 GPUs, ~1 h 25 min in total. LLaMAFactory's preprocessing decodes every clip to count tokens
  (~11 min per 1294-row shard) and node speed varied 1.5–3×.
- **Restored items** are part of this model's main shard files (`eval/preds/egoavu_r100k/`), not a separate `restored_v3/`
  directory; `latency_s` and `n_input_tokens` are not recorded for this model.

## 10. Other notes

- **gpt-6-astra** was not run. Through the gateway it accepts text and images only (`video_url` and `input_audio` are rejected),
  costs $75 / M input and $150 / M output tokens, and uses ~154 tokens per 512-px frame; a full run would cost roughly $400–
  $7,800 depending on how many frames are sent, and it would be a vision-only baseline.
- **Operational:** `pkill -f` / `pgrep -f` inside `ssh 'bash -c …'` matches the wrapper's own command line and killed the
  session twice — patterns are anchored with `^` now. Long jobs are started with `setsid nohup`; one job launched in the
  foreground lost its parent when the SSH connection dropped (its workers survived). An earlier training job on the GPUs was
  stopped for the evaluation with the owner's agreement.

## 11. H cluster, 2026-09-24/25: full fine-tune, EgoTaskQA and the EgoToM-kit re-test

- **Thinker-only full fine-tune would not load.** `ckpt_fft_epoch2` is saved as `qwen2_5_omni_thinker`
  (`Qwen2_5OmniThinkerForConditionalGeneration`, tensor names without `thinker.`); transformers 4.54's Auto mappings and the
  LLaMA-Factory recipe only know the full `qwen2_5_omni` model. **Handled:** `h_cluster/convert_thinker_ckpt.py` re-headers the
  safetensors shards with the `thinker.` prefix and copies the bytes verbatim (data sha256 verified, all 1,346 tensors checked against
  the base thinker), using the base's thinker-view config and tokenizer/processor files (the checkpoint's own differ only in
  serialization/training fields). A smoke job (EgoCross + EgoSchema) had to log a clean load before the other benchmarks were submitted.
- **GPUs idle while jobs waited.** 1-GPU jobs requesting 24 CPUs / 192 GB sat Inqueue with "Insufficient cpu" on the nodes that still
  had free GPUs (their CPUs were largely taken by our own 24-CPU jobs). **Handled:** request the per-GPU share (8 CPU / 64 GB) and run
  3–4 bs=1 processes per H200 (~26 GB each).
- **Uneven shard tails.** With 30 shards the slowest EgoAVU-Bench shard would have finished ~40 min after the rest. **Handled:** the
  unfinished rows were redistributed (longest windows first, snake order) over 64 fresh shards and merged back
  (`split_kit_rest.py` / `finalize_kit_rest.py`); a chain restarted with "attempts = 3" markers still resubmitted the stopped jobs once
  (the name it checks did not exist yet) — they were stopped within a minute and the base was removed from that chain's loop.
- **egoOmni under the kit protocol (dropped).** The kit instances were first launched without their data files; then decord could
  not decode the egoOmni source clips and `qwen_omni_utils` fell back to `torchvision.io.read_video`, which loads a whole clip at full
  resolution (~17 GB for 90 s of 1080p) and got the 4-process jobs OOM-killed. `nat_predict_lf.py` now falls back to a PyAV reader
  with the same frame selection, and 2 fps proxies were being prepared, when the user decided egoOmni does not need this re-test; no
  kit-protocol egoOmni numbers are reported.
- **Short windows.** `qwen_omni_utils` needs ≥ 2 frames; 2 EgoAVU-Bench windows (0.125 s, 0.94 s) were widened to 1 s for the video
  under the kit protocol (audio keeps the original window).
