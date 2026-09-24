#!/usr/bin/env python
"""Join LLaMAFactory generated_predictions.jsonl to the dataset index (order) — verified: label == gold of the asked turn
and the prompt contains the asked question — and append harness-schema rows to out/s{k}of{K}.jsonl (resumable source of
truth; self-round building reads it). With --final, also writes eval/preds/egoavu_r20k32g/shard{k}.jsonl (+ self round-1 copies).
  collect.py DATASET K k [--final]"""
import json, os, sys
sys.path.insert(0, "/ai4good1-shared/liyu/egoOmni_baselines/eval")
from egoomni_eval.data import request_for, read_jsonl
from build_data import R, plans

name, K, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
final = "--final" in sys.argv
TAG, MODEL = "egoavu_r20k32g", "/ai4good1-shared/liyu/egoavu/release/EgoAVU-Qwen2.5-Omni-7B-LoRA-r20k/32gpu"
P = {p.item_id: p for p in plans()}
out_p = f"{R}/out/s{k}of{K}.jsonl"; os.makedirs(f"{R}/out", exist_ok=True)
prev = read_jsonl(out_p)
if name != "none":
    idx = json.load(open(f"{R}/index/{name}.json"))
    preds = read_jsonl(f"{R}/runs/{name}/generated_predictions.jsonl")
    assert len(preds) == len(idx), f"{len(preds)} predictions vs {len(idx)} rows"
    have = {(r["item_id"], r["turn_idx"], r["protocol"]) for r in prev}
    own = {(r["item_id"], r["turn_idx"]): r["pred"] for r in prev if r["protocol"] == "self" or (r["protocol"] == "gold" and r["turn_idx"] == 1)}
    bad, new = [], []
    for (iid, t, proto), pr in zip(idx, preds):
        p = P[iid]
        if pr["label"].strip() != p.turns[t - 1]["gold"].strip() or p.turns[t - 1]["question"].strip()[:60] not in pr["prompt"]:
            bad.append((iid, t))
        hist = [p.turns[j]["gold"] for j in range(t - 1)] if proto == "gold" else [own.get((iid, j + 1)) for j in range(t - 1)]
        r = request_for(p, t, proto, hist).to_row()
        r.update(model_tag=TAG, model_dir=MODEL, pred=pr["predict"].strip(), error=None, modality_used="av" if p.has_audio else "v",
                 n_input_tokens=None, latency_s=None, prep_wait_s=None, llamafactory_dataset=name)
        if (iid, t, proto) not in have:
            new.append(r)
    assert not bad, f"join verification failed for {len(bad)} rows, e.g. {bad[:3]}"
    with open(out_p, "a") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: joined {len(preds)} (verified), appended {len(new)}")
if final:
    rows = read_jsonl(out_p)
    g1 = {r["item_id"]: r for r in rows if r["protocol"] == "gold" and r["turn_idx"] == 1 and P[r["item_id"]].multi}
    selfkeys = {(r["item_id"], r["turn_idx"]) for r in rows if r["protocol"] == "self"}
    for iid, r in g1.items():            # self round 1 == gold round 1 (same convention as run_infer.py)
        if (iid, 1) not in selfkeys:
            rows.append(dict(r, protocol="self"))
    d = f"/ai4good1-shared/liyu/egoOmni_baselines/eval/preds/{TAG}"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/shard{k}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"final → {d}/shard{k}.jsonl rows={len(rows)}")
