#!/usr/bin/env python3
"""Paper-style AVH accuracy over ALL items per subtype (196 / 184 / 196), which is what the paper's Table 4 denominators imply.
Yes/no probes are scored by extraction (closed_ended_acc.yn_extract); the 272 open-ended "What..." items need a right/wrong rule the
paper does not state. Candidate rules: open item correct iff the official judge rating >= t (t = 2..5). The rule is chosen on the
BASE model control by matching the paper's base Qwen2.5-Omni-7B values (Action 44.39 = 87/196, Object 50.00 = 92/184,
Sound 33.67 = 66/196), then applied unchanged to every model.
usage: avh_calibrate.py TAG=PRED_JSON=JUDGE_DIR [...]   (first TAG = the base control used for calibration)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_ended_acc as cea
PAPER_BASE = {"Action": 44.39, "Object": 50.00, "Sound": 33.67}
specs = [a.split("=") for a in sys.argv[1:]]
out = {}
for tag, pred, jdir in specs:
    preds = {p["bench_idx"]: p for p in json.load(open(pred))}
    rating = {x["bench_idx"]: x["rating"] for x in map(json.loads, open(f"{jdir}/judge_items.jsonl")) if x["file_name"] == os.path.basename(pred)}
    res = {}
    for t in (2, 3, 4, 5):
        sub = {}
        for st in ("Action", "Object", "Sound"):
            cat = f"Audio Visual Hallucination ({st})"; ok = n = 0
            for i, r in enumerate(cea.rows):
                if r["category"] != cat:
                    continue
                n += 1; g = cea.YN.match(r["answer"])
                if g:
                    ok += cea.yn_extract(preds[i]["output"])[0] == g.group(1).lower()
                else:
                    ok += (rating[i] or 0) >= t
            sub[st] = {"acc": round(100 * ok / n, 2), "k": ok, "n": n}
        macro = round(sum(v["acc"] for v in sub.values()) / 3, 2)
        err = round(sum(abs(sub[s]["acc"] - PAPER_BASE[s]) for s in sub) / 3, 2)
        res[f"open_correct_if_judge>={t}"] = {"by_subtype": sub, "macro": macro, "mean_abs_diff_to_paper_base": err}
    out[tag] = res
print(json.dumps(out, indent=1))
