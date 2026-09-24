#!/usr/bin/env python3
"""Merge the rebalanced tail back (see split_kit_rest.py): once EVERY rest shard has all rows ok, each unfinished original shard
egoavu_bench_eval_s{k}of30 gets its generated_predictions.jsonl (its own ok rows + the rest rows, in order) and predict_results.json,
then the unchanged collect_predictions.py (label / question-in-prompt checks) writes collect_stats.json. Idempotent; run at the end
of every rest job, so the last one to finish completes it.  usage: finalize_kit_rest.py R"""
import json, os, subprocess, sys
BI = "/ai4good1-shared/liyu/egoavu/bench_infer"; R = int(sys.argv[1]); TAG = "kit_base"


def ok_rows(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("status") == "ok":
                out[r["i"]] = r
    return out


merged = {}
for j in range(R):
    rows = [json.loads(l) for l in open(f"{BI}/data/kitrest_s{j}of{R}.jsonl")]
    ok = ok_rows(f"{BI}/runs/kit_base_rest/kitrest_s{j}of{R}/nat_rows.jsonl")
    if len(ok) < len(rows):
        print(f"rest shard {j}: {len(ok)}/{len(rows)} done - not finalizing yet"); sys.exit(0)
    for i, row in enumerate(rows):
        merged[(row["_orig_shard"], row["_orig_i"])] = ok[i]
for k in range(30):
    d = f"{BI}/runs/{TAG}/egoavu_bench_eval_s{k}of30"
    if os.path.exists(f"{d}/collect_stats.json"):
        continue
    n = sum(1 for _ in open(f"{BI}/data/egoavu_bench_eval_s{k}of30.jsonl"))
    done = ok_rows(f"{d}/nat_rows.jsonl")
    done.update({i: r for (kk, i), r in merged.items() if kk == k})
    missing = [i for i in range(n) if i not in done]
    assert not missing, f"shard {k}: rows {missing[:5]} have no result"
    with open(f"{d}/generated_predictions.jsonl.tmp", "w") as f:
        for i in range(n):
            f.write(json.dumps({x: done[i][x] for x in ("prompt", "predict", "label", "n_video_tokens", "n_audio_tokens")}, ensure_ascii=False) + "\n")
    os.replace(f"{d}/generated_predictions.jsonl.tmp", f"{d}/generated_predictions.jsonl")
    json.dump({"predict_runtime": round(sum(done[i]["seconds"] for i in range(n)), 1), "predict_samples": n,
               "protocol": "colleague EgoToM kit (nat_predict_lf.py); tail rebalanced over kitrest shards", "max_new_tokens": 1024,
               "max_frames": 360}, open(f"{d}/predict_results.json", "w"), indent=2)
    subprocess.run([sys.executable, f"{BI}/tools/collect_predictions.py", TAG, f"egoavu_bench_eval_s{k}of30"], cwd="/tmp", check=True,
                   stdout=subprocess.DEVNULL)
    print(f"shard {k}: finalized ({n} rows)")
print("finalize done")
