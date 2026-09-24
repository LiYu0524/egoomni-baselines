#!/usr/bin/env python3
"""Copy preds/<tag>/ to preds_final/<tag>/ keeping only the FINAL egoOmni bench: all original items + the 117 restored_v3
items listed in data/restored_v3/final_bench_restored_ids.json (24 evaluated restored items were dropped from the final set).
usage: filter_final_preds.py TAG"""
import glob, json, os, sys
E = "/ai4good1-shared/liyu/egoOmni_baselines/eval"; tag = sys.argv[1]
keep = set(json.load(open(f"{E}/data/restored_v3/final_bench_restored_ids.json"))["sample_ids"])
n_in = n_out = 0
for src in glob.glob(f"{E}/preds/{tag}/**/*.jsonl", recursive=True):
    dst = src.replace(f"{E}/preds/", f"{E}/preds_final/"); os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src) as f, open(dst, "w") as g:
        for l in f:
            n_in += 1; iid = json.loads(l)["item_id"]
            if not iid.startswith("restored_v3:") or iid in keep:
                g.write(l); n_out += 1
print(f"[final] {tag}: kept {n_out}/{n_in} rows")
