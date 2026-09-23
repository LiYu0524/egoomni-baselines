#!/usr/bin/env python
"""Join generated_predictions.jsonl back to the bench rows (order) with verification — same checks and output schema as
bench_infer/tools/collect_predictions.py: prediction label == gold answer, prompt contains the question.
usage: collect_fft.py TAG DATASET"""
import json, sys
from collections import Counter
tag, dataset = sys.argv[1:3]
BI = "/ai4good1-shared/liyu/egoavu/bench_infer"; R = "/ai4good1-shared/liyu/egoavu/bench_infer_fft"
rows = [json.loads(l) for l in open(f"{BI}/data/{dataset}.jsonl")]
out_dir = f"{R}/runs/{tag}/{dataset}"
preds = [json.loads(l) for l in open(f"{out_dir}/generated_predictions.jsonl")]
assert len(preds) == len(rows), f"{len(preds)} predictions vs {len(rows)} rows"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/ai4good1-shared/liyu/egoavu/models/Qwen2.5-Omni-7B")
items, bad = [], []
for r, p in zip(rows, preds):
    q = r["messages"][0]["content"].replace("<video>", "").replace("<audio>", ""); a = r["messages"][1]["content"]
    if p["label"].strip() != a.strip() or q.strip() not in p["prompt"]:
        bad.append(r["bench_idx"])
    n_tok = len(tok(p["predict"], add_special_tokens=False)["input_ids"])
    items.append({"bench_idx": r["bench_idx"], "video_id": r["video_id"], "start_time": r["start_time"], "end_time": r["end_time"],
                  "category": r["category"], "category_short": r["category_short"], "question": q, "answer": a,
                  "output": p["predict"], "output_tokens": n_tok, "truncated": n_tok >= 1024, "audio_used": bool(r["audios"]),
                  "model": tag})
assert not bad, f"join verification failed for {len(bad)} rows, e.g. bench_idx {bad[:5]}"
with open(f"{out_dir}/predictions.jsonl", "w") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
st = {"tag": tag, "dataset": dataset, "rows": len(items), "join_verified": True,
      "empty_outputs": sum(not it["output"].strip() for it in items), "truncated": sum(it["truncated"] for it in items),
      "categories": Counter(it["category"] for it in items)}
json.dump(st, open(f"{out_dir}/collect_stats.json", "w"), indent=2); print(json.dumps(st))
