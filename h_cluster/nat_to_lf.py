#!/usr/bin/env python3
"""Put nat_infer.py results where the existing scorers read LLaMAFactory output: <kit>/runs/<TAG>/<dataset>/
generated_predictions.jsonl, rows in dataset order with the fields the scorers use (predict, label; prompt left empty).
Refuses unless every request of the benchmark has an ok result and each echoed label matches the dataset row.
usage: nat_to_lf.py BENCH MODEL TAG      e.g. nat_to_lf.py egoschema ckpt_epoch2 kit_ckpt_epoch2"""
import glob, json, os, sys

L = "/ai4good1-shared/liyu"
KIT = {"egoschema": "egoschema_eval_h", "egotaskqa": "egotaskqa_eval_h", "egocross": "egocross_eval_h"}
bench, model, tag = sys.argv[1:4]
reqs = [json.loads(l) for l in open(f"{L}/natkit/requests/{bench}.jsonl")]
res = {}
for f in sorted(glob.glob(f"{L}/natkit/out/{bench}/{model}/result_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r["status"] == "ok":
            res[r["sample_id"]] = r
missing = [q["sample_id"] for q in reqs if q["sample_id"] not in res]
assert not missing, f"{len(missing)} of {len(reqs)} requests have no ok result, e.g. {missing[:3]}"
by_ds = {}
for q in reqs:
    r = res[q["sample_id"]]
    assert r["label"] == q["label"], q["sample_id"]
    by_ds.setdefault(q["dataset"], []).append((q["row"], {"prompt": "", "predict": r["pred"], "label": q["label"]}))
for ds, rows in by_ds.items():
    rows.sort(key=lambda x: x[0])
    assert [i for i, _ in rows] == list(range(len(rows))), ds
    d = f"{L}/{KIT[bench]}/runs/{tag}/{ds}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/generated_predictions.jsonl", "w") as f:
        for _, g in rows:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
hit = sum(1 for r in res.values() if r.get("hit_max_new_tokens"))
aud = sum(1 for r in res.values() if r.get("use_audio_in_video"))
print(f"{bench}/{model} -> runs/{tag}: {len(reqs)} predictions in {len(by_ds)} datasets | with audio {aud} | hit max_new_tokens {hit} "
      f"| median input tokens {sorted(r['input_tokens'] for r in res.values())[len(res) // 2]}")
