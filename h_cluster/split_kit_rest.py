#!/usr/bin/env python3
"""Rebalance the unfinished tail of the kit-protocol EgoAVU-Bench base run: every row of egoavu_bench_eval_s{k}of30 without an ok
row in runs/kit_base/.../nat_rows.jsonl goes to one of R rest shards (longest windows first, snake order, so shards cost about the
same). Rows keep their full content plus _orig_shard / _orig_i for the merge back (finalize_kit_rest.py).  usage: split_kit_rest.py R"""
import json, os, sys
BI = "/ai4good1-shared/liyu/egoavu/bench_infer"; R = int(sys.argv[1])
rest = []
for k in range(30):
    d = f"{BI}/runs/kit_base/egoavu_bench_eval_s{k}of30"
    if os.path.exists(f"{d}/collect_stats.json"):
        continue
    ok = set()
    if os.path.exists(f"{d}/nat_rows.jsonl"):
        for line in open(f"{d}/nat_rows.jsonl"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok":
                ok.add(r["i"])
    for i, line in enumerate(open(f"{BI}/data/egoavu_bench_eval_s{k}of30.jsonl")):
        if i not in ok:
            rest.append(dict(json.loads(line), _orig_shard=k, _orig_i=i))
rest.sort(key=lambda r: -(r["end_time"] - r["start_time"]))
buckets = [[] for _ in range(R)]
for n, r in enumerate(rest):
    b = n % (2 * R)
    buckets[b if b < R else 2 * R - 1 - b].append(r)
for j, b in enumerate(buckets):
    with open(f"{BI}/data/kitrest_s{j}of{R}.jsonl", "w") as f:
        for r in b:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rest)} unfinished rows -> {R} rest shards of {min(map(len, buckets))}-{max(map(len, buckets))} rows")
