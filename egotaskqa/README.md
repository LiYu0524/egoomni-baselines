# EgoTaskQA — Qwen2.5-Omni-7B models (2026-09-25)

The EgoTaskQA **test** splits, open short-answer questions about egocentric task videos (LEMMA recordings):

| split | questions | what it tests |
|---|---|---|
| **direct** | 8,783 | questions name the objects / actions directly |
| **indirect** | 10,153 | the same kinds of questions, but objects / actions are referred to through other events ("the object used before …"), testing compositional reference |

Source: `s3://safevlagent/liyu/benchmarks/egotaskqa/data/` (the official release: `qa_videos.zip` + `qa_{direct,indirect}_test_qas.json`);
dataset files are not redistributed here.

## Protocol (Table 1 of the top-level README)
- **Media** (`prep_media.py`, CPU job): every test clip streamed out of `qa_videos.zip` and turned into the proxy format the EgoAVU LoRAs were
  trained on — 2 fps, ~262k px/frame, H.264; the clips have no audio track, so no audio is fed. 2,234 clips, median 33 s, max 167 s.
- **Rows** (`build_data.py`): the question, then `Answer the question using a single word or phrase.`; video = the whole proxy
  (`<proxy>.mp4#t=0,<dur>`); each split sharded 4-way.
- **Inference** (`make_config.py`, `../h_cluster/egotaskqa_h.sh`): LLaMA-Factory predict, EgoAVU training media recipe (2 fps, ≤ 64 frames,
  ≤ 200,704 px), `qwen2_omni` template, greedy, 32 new tokens. `ADAPTER` = a LoRA dir, `none` (base) or `model=DIR` (a full-weight model).
- **Scoring** (`score_judge.py`, `../h_cluster/egotaskqa_score_h.sh`): predictions are joined back by order (the echoed label must equal the
  gold answer), then two accuracies per split and per EgoTaskQA breakdown (type / category / semantic / structural):
  - **EM** — normalized exact match: lower-cased, punctuation and the articles a / an / the removed; for yes/no questions only the first
    word of the prediction is compared, otherwise the whole normalized answer must equal the normalized reference.
  - **judge** — EM, plus for every non-yes/no answer that EM rejected, Qwen3-32B (vLLM TP1, thinking off, greedy, max 4 tokens) is asked
    whether the predicted answer means the same thing as the reference for this question ("synonyms and paraphrases are fine; a different
    object, action, state or attribute is wrong"; one-word yes/no reply). Yes/no questions keep the EM decision, so **judge ≥ EM**.
    EgoTaskQA was designed for classification over a fixed answer vocabulary; generative models phrase short answers freely
    ("chopping board" vs "cutting board"), which EM alone under-counts. Judge replies were parseable for every item.

The **EgoToM-kit protocol** re-test (Table 2) feeds the same rows through `../h_cluster/nat_infer.py` (native transformers,
`qwen_omni_utils`, all frames at 2 fps — median 66, max 334 — at ≤ 156,800 px, 256 new tokens) and scores them with the same `score_judge.py`.

## Results (`results/<tag>/scores.json`; per-question outputs incl. judge replies in `results/<tag>/outputs.jsonl.gz`)

| model | direct EM | direct judge | indirect EM | indirect judge |
|---|---|---|---|---|
| base (untuned Qwen2.5-Omni-7B) | **29.96** | **34.52** | **34.62** | 38.17 |
| r100k (ours) | 22.16 | 29.50 | 27.86 | 33.28 |
| r20k-8gpu (ours) | 25.06 | 30.88 | 29.77 | 34.17 |
| r20k-32gpu (ours) | 24.50 | 30.45 | 29.35 | 34.12 |
| ckpt_sft (colleague LoRA) | 25.65 | 33.31 | 31.02 | 36.81 |
| ckpt_epoch2 = ckpt_lora_epoch2 (colleague LoRA) | 24.52 | 33.71 | 31.04 | **38.65** |
| ckpt_fft_epoch2 (colleague full FT) | 23.10 | 32.62 | 29.07 | 36.41 |
| base, kit protocol | 30.07 | 34.41 | 34.60 | 38.33 |
| ckpt_epoch2, kit protocol | 24.42 | 33.58 | 30.84 | 38.33 |
| ckpt_fft_epoch2, kit protocol | 23.22 | 32.64 | 28.99 | 36.26 |

Yes/no vs open-answer splits and every breakdown are in the `scores.json` files; the top-level README has the yes/no and open columns.
