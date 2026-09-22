#!/usr/bin/env python
"""Split the restored_v3 items into N duration-balanced subsets for parallel jobs → subsets/restored_v3_part{k}of{N}.json"""
import json, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
items = json.load(open("/ai4good1-shared/liyu/egoOmni/test/qa_restored_v3.json"))
iid = lambda it: it.get("qa_id") or it.get("sample_id")
parts, load = [[] for _ in range(N)], [0.0] * N
for it in sorted(items, key=lambda x: -x["clip_duration_sec"]):
    k = load.index(min(load)); parts[k].append(iid(it)); load[k] += it["clip_duration_sec"] + 20
for k, p in enumerate(parts):
    json.dump({"item_ids": sorted(p)}, open(f"subsets/restored_v3_part{k}of{N}.json", "w"))
print("parts:", [len(p) for p in parts], "load(s):", [round(x) for x in load])
