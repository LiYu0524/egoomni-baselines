#!/usr/bin/env python3
"""Which kit-protocol shards are complete: prints one line "<bench> <model> <shard>" per shard whose every request
(requests[shard::NUM_SHARDS]) has an ok row in natkit/out/<bench>/<model>/result_<shard>.jsonl.
usage: nat_status.py BENCH:NUM_SHARDS [BENCH:NUM_SHARDS ...]"""
import json, os, sys

K = "/ai4good1-shared/liyu/natkit"
MODELS = ("base", "ckpt_epoch2", "ckpt_fft_epoch2")
for spec in sys.argv[1:]:
    bench, ns = spec.split(":"); ns = int(ns)
    ids = [json.loads(l)["sample_id"] for l in open(f"{K}/requests/{bench}.jsonl")]
    for m in MODELS:
        for s in range(ns):
            p = f"{K}/out/{bench}/{m}/result_{s}.jsonl"
            if not os.path.exists(p):
                continue
            ok = set()
            for line in open(p):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("status") == "ok":
                    ok.add(r["sample_id"])
            if all(i in ok for i in ids[s::ns]):
                print(bench, m, s)
