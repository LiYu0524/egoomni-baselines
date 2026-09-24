#!/usr/bin/env python3
"""Assemble the judge-ready release: one JSON list per model in official bench order (bench_idx 0..3975), readable by the
official EgoAVU evaluation/llm_as_judge.py and captioning_eval.py (fields category / question / answer / output).
Adds exact per-row input facts: video/audio token counts parsed from the prompt the model actually received, and window
media flags from tools/window_media_flags.py. JSON is written ASCII-only (ensure_ascii) so it loads under any locale.
usage: assemble_release.py OUT_DIR TAG [TAG ...]"""
import json, os, sys, statistics
from collections import Counter
from transformers import AutoTokenizer

BI = "/ai4good1-shared/liyu/egoavu/bench_infer"
out, tags = sys.argv[1], sys.argv[2:]
for d in ("predictions", "stats", "notes"):
    os.makedirs(f"{out}/{d}", exist_ok=True)
tok = AutoTokenizer.from_pretrained("/ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B")
data = [json.loads(l) for l in open(f"{BI}/data/egoavu_bench_eval.jsonl")]
flags = {int(k): v for k, v in json.load(open(f"{BI}/data/window_media_flags.json")).items()}
gold_tok = {r["bench_idx"]: len(tok(r["messages"][1]["content"], add_special_tokens=False)["input_ids"]) for r in data}
dur = {r["bench_idx"]: r["end_time"] - r["start_time"] for r in data}

for tag in tags:
    rd = f"{BI}/runs/{tag}/egoavu_bench_eval"
    gen = [json.loads(l) for l in open(f"{rd}/generated_predictions.jsonl")]
    assert len(gen) == len(data)
    counts = {}
    for r, g in zip(data, gen):                     # same order; label check re-done here as a guard
        assert g["label"].strip() == r["messages"][1]["content"].strip(), r["bench_idx"]
        # native (EgoToM-kit) runs record the counts measured on input_ids; LLaMAFactory prompts carry the expanded tokens
        counts[r["bench_idx"]] = (g["n_video_tokens"], g["n_audio_tokens"]) if "n_video_tokens" in g else \
            (g["prompt"].count("<|VIDEO|>"), g["prompt"].count("<|AUDIO|>"))
    rows = [json.loads(l) for l in open(f"{rd}/predictions.jsonl")]
    rows.sort(key=lambda r: r["bench_idx"])
    assert [r["bench_idx"] for r in rows] == list(range(3976)), tag
    for r in rows:
        nv, na = counts[r["bench_idx"]]; f = flags[r["bench_idx"]]
        r.update(n_video_tokens=nv, n_audio_tokens=na, audio_seconds_fed=round(na / 25, 2),
                 audio_capped_at_300s=bool(na == 7500 and dur[r["bench_idx"]] > 300),
                 video_black_frames_of_8=f["black_frames_of_8"], video_all_black=f["video_all_black"],
                 audio_silent=f["audio_silent"])
        assert (na > 0) == r["audio_used"], r["bench_idx"]
    with open(f"{out}/predictions/{tag}.json", "w") as fh:
        json.dump(rows, fh, ensure_ascii=True, indent=1)
    cats = sorted({r["category"] for r in rows})
    st = {"tag": tag, "rows": len(rows), "join_verified": True,
          "empty_outputs": sum(not r["output"].strip() for r in rows),
          "truncated_at_1024_tokens": [r["bench_idx"] for r in rows if r["truncated"]],
          "output_vs_gold_tokens_by_category": {c: {
              "n": sum(r["category"] == c for r in rows),
              "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows if r["category"] == c), 1),
              "mean_gold_tokens": round(statistics.mean(gold_tok[r["bench_idx"]] for r in rows if r["category"] == c), 1)} for c in cats},
          "predict_runtime_s": json.load(open(f"{rd}/predict_results.json"))["predict_runtime"],
          "categories": Counter(r["category"] for r in rows)}
    json.dump(st, open(f"{out}/stats/{tag}.json", "w"), indent=2)
    print(tag, len(rows), "truncated", len(st["truncated_at_1024_tokens"]))

# model-independent notes on bench windows (bench_idx lists)
fl = lambda cond: [i for i in range(3976) if cond(i)]
notes = {
    "video_all_black": fl(lambda i: flags[i]["video_all_black"]),
    "video_partly_black": fl(lambda i: 0 < flags[i]["black_frames_of_8"] < 8),
    "audio_silent": fl(lambda i: flags[i]["audio_silent"]),
    "window_shorter_than_5s": fl(lambda i: dur[i] < 5),
    "no_audio_fed": [r["bench_idx"] for r in data if not r["audios"]],
    "audio_capped_at_300s": fl(lambda i: dur[i] > 300),
}
notes["_counts"] = {k: len(v) for k, v in notes.items()}
json.dump(notes, open(f"{out}/notes/bench_window_notes.json", "w"), indent=1)
print(notes["_counts"])
