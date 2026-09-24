#!/usr/bin/env python3
"""Merge the K interleaved EgoAVU-Bench shard runs of a tag (shard k = rows with bench_idx % K == k) into
runs/<tag>/egoavu_bench_eval/generated_predictions.jsonl in the full dataset's row order; collect_predictions.py then
re-joins and re-verifies it exactly as for a single full run.  usage: merge_bench_shards.py TAG [K]"""
import json, os, sys
BI = "/ai4good1-shared/liyu/egoavu/bench_infer"
tag = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
full = [json.loads(l)["bench_idx"] for l in open(f"{BI}/data/egoavu_bench_eval.jsonl")]
gen = {}
for k in range(K):
    ds = f"egoavu_bench_eval_s{k}of{K}"
    ids = [json.loads(l)["bench_idx"] for l in open(f"{BI}/data/{ds}.jsonl")]
    g = [l.rstrip("\n") for l in open(f"{BI}/runs/{tag}/{ds}/generated_predictions.jsonl")]
    assert len(g) == len(ids), f"{ds}: {len(g)} predictions vs {len(ids)} rows"
    gen.update(zip(ids, g))
assert sorted(gen) == sorted(full) and len(gen) == len(full), "shards do not cover the bench exactly once"
out = f"{BI}/runs/{tag}/egoavu_bench_eval"; os.makedirs(out, exist_ok=True)
with open(f"{out}/generated_predictions.jsonl", "w") as f:
    for i in full:
        f.write(gen[i] + "\n")
rt = [json.load(open(f"{BI}/runs/{tag}/egoavu_bench_eval_s{k}of{K}/predict_results.json")) for k in range(K)]
json.dump({"predict_runtime": round(sum(r["predict_runtime"] for r in rt), 1), "predict_samples": len(full),
           "note": f"sum over {K} shard runs (2 bs=1 processes per H200); wall time per shard max "
                   f"{max(r['predict_runtime'] for r in rt):.0f} s"}, open(f"{out}/predict_results.json", "w"), indent=2)
print(f"[merge] {tag}: {len(full)} rows from {K} shards -> {out}")
