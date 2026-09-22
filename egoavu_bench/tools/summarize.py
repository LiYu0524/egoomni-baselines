#!/usr/bin/env python3
"""Build results/SUMMARY.md + summary.json for EgoAVU-Bench from the scored runs.
usage: summarize.py OUT_DIR TAG=PRED_JSON=JUDGE_DIR [TAG=PRED_JSON=JUDGE_DIR ...]
JUDGE_DIR holds judge_items.jsonl (official judge, all 7 categories) and caption_audiovisual{segment,dense}narration.csv (official
captioning_eval.py). Closed-ended accuracy is recomputed with tools/closed_ended_acc.py. Bootstrap: 2,000 resamples, seed 0."""
import csv, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_ended_acc as cea

out_dir = sys.argv[1]; specs = [a.split("=") for a in sys.argv[2:]]
os.makedirs(out_dir, exist_ok=True)
CATS = {"SSA": "Sound Source Association", "AVDN": "Audio-Visual Dense Narration", "AVSN": "Audio-Visual Segment Narration",
        "TR": "Temporal Reasoning", "AVH-Action": "Audio Visual Hallucination (Action)",
        "AVH-Object": "Audio Visual Hallucination (Object)", "AVH-Sound": "Audio Visual Hallucination (Sound)"}
PAPER = {"paper: Qwen2.5-Omni-7B (base)": dict(SSA=1.50, AVDN_S=2.37, AVDN_M=10.69, AVDN_R=14.74, AVSN_S=1.99, AVSN_M=9.99, AVSN_R=13.39, TR=53.20, AVH=42.69),
         "paper: Ours (LoRA)": dict(SSA=3.15, AVDN_S=2.60, AVDN_M=12.20, AVDN_R=17.19, AVSN_S=2.45, AVSN_M=22.53, AVSN_R=28.34, TR=64.31, AVH=61.69)}
rng = random.Random(0); B = 2000
def mean(v): return sum(v) / len(v)
def ci(v):
    n = len(v); bs = sorted(mean([v[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return round(bs[int(.025 * B)], 3), round(bs[int(.975 * B)], 3)

res, per_item = {}, {}
for tag, pred, jdir in specs:
    fn = os.path.basename(pred)
    items = [json.loads(l) for l in open(f"{jdir}/judge_items.jsonl")]
    items = [x for x in items if x["file_name"] == fn]
    assert len(items) == 3976, (tag, len(items))
    rating = {x["bench_idx"]: x["rating"] for x in items}
    preds = {p["bench_idx"]: p for p in json.load(open(pred))}
    cap = {}
    for key, fcsv in (("AVSN", "caption_audiovisualsegmentnarration.csv"), ("AVDN", "caption_audiovisualdensenarration.csv")):
        row = next(r for r in csv.DictReader(open(f"{jdir}/{fcsv}")) if r["file_name"] == fn)
        cap[key] = (float(row["Meteor"]), float(row["Rouge"]))
    closed = cea.score(pred)
    per_cat = {}
    for k, c in CATS.items():
        v = [rating[i] for i, r in enumerate(cea.rows) if r["category"] == c and rating[i] is not None]
        per_cat[k] = {"n": len(v), "S": round(mean(v), 3), "ci95": ci(v)}
    # closed/open split of judge S for TR and AVH (supplementary)
    closed_idx = {i for i, r in enumerate(cea.rows) if (r["category"] == "Temporal Reasoning" and cea.GOLD_MCQ.match(r["answer"])) or
                  (r["category"].startswith("Audio Visual Hallucination") and cea.YN.match(r["answer"]))}
    split = {}
    for k in ("TR", "AVH-Action", "AVH-Object", "AVH-Sound"):
        c = CATS[k]
        for part, cond in (("closed", lambda i: i in closed_idx), ("open", lambda i: i not in closed_idx)):
            v = [rating[i] for i, r in enumerate(cea.rows) if r["category"] == c and cond(i)]
            split[f"{k}/{part}"] = {"n": len(v), "S": round(mean(v), 3)}
    # accuracies with CI from per-item correctness
    tr_c, avh_c = [], {"Action": [], "Object": [], "Sound": []}
    for i, r in enumerate(cea.rows):
        g = cea.GOLD_MCQ.match(r["answer"])
        if r["category"] == "Temporal Reasoning" and g:
            opts = dict(cea.OPT_LINE.findall(r["question"])); tr_c.append(float(cea.mcq_extract(preds[i]["output"], opts)[0] == g.group(1)))
        if r["category"].startswith("Audio Visual Hallucination") and cea.YN.match(r["answer"]):
            avh_c[r["category"].split("(")[1].rstrip(")")].append(float(cea.yn_extract(preds[i]["output"])[0] == cea.YN.match(r["answer"]).group(1).lower()))
    avh_macro = mean([100 * mean(v) for v in avh_c.values()])
    yes_rate = mean([1.0 if cea.yn_extract(preds[i]["output"])[0] == "yes" else 0.0 for i, r in enumerate(cea.rows)
                     if r["category"].startswith("Audio Visual Hallucination") and cea.YN.match(r["answer"])])
    ratio = {}
    for k in ("AVSN", "AVDN", "SSA", "TR"):
        rs = sorted(len(preds[i]["output"].split()) / max(1, len(r["answer"].split())) for i, r in enumerate(cea.rows) if r["category"] == CATS[k])
        ratio[k] = round(rs[len(rs) // 2], 2)
    res[tag] = {"SSA": per_cat["SSA"]["S"], "AVDN_S": per_cat["AVDN"]["S"], "AVDN_M": cap["AVDN"][0], "AVDN_R": cap["AVDN"][1],
                "AVSN_S": per_cat["AVSN"]["S"], "AVSN_M": cap["AVSN"][0], "AVSN_R": cap["AVSN"][1],
                "TR": round(100 * mean(tr_c), 2), "TR_ci95": [round(100 * x, 1) for x in ci(tr_c)], "TR_n": len(tr_c),
                "AVH": round(avh_macro, 2), "AVH_by_subtype": {k: round(100 * mean(v), 2) for k, v in avh_c.items()},
                "AVH_n": sum(len(v) for v in avh_c.values()), "AVH_yes_rate": round(100 * yes_rate, 1),
                "judge_S_all_categories": per_cat, "judge_S_closed_open_split": split, "median_output_to_gold_word_ratio": ratio,
                "closed_ended_extraction": {"TR": closed["TR_mcq"]["extraction"], "AVH": closed["AVH_yesno"]["extraction"]},
                "judge_unparseable": sum(1 for x in items if x["rating"] is None)}
    per_item[tag] = (rating, tr_c, avh_c)

# paired differences between consecutive tags (same items)
pairs = {}
tags = [s[0] for s in specs]
for a in range(len(tags)):
    for b in range(a + 1, len(tags)):
        ta, tb = tags[a], tags[b]; ra, rb = per_item[ta][0], per_item[tb][0]; d = {}
        for k, c in CATS.items():
            diff = [rb[i] - ra[i] for i, r in enumerate(cea.rows) if r["category"] == c and ra[i] is not None and rb[i] is not None]
            d[k] = {"mean_diff": round(mean(diff), 3), "ci95": ci(diff)}
        diff = [y - x for x, y in zip(per_item[ta][1], per_item[tb][1])]; d["TR_acc_pts"] = {"mean_diff": round(100 * mean(diff), 2), "ci95": [round(100 * x, 1) for x in ci(diff)]}
        pairs[f"{tb} - {ta}"] = d

json.dump({"models": res, "paired_differences": pairs, "paper_reference": PAPER}, open(f"{out_dir}/summary.json", "w"), indent=1)
cols = ["SSA", "AVDN_S", "AVDN_M", "AVDN_R", "AVSN_S", "AVSN_M", "AVSN_R", "TR", "AVH"]
L = ["| model | SSA S | AVDN S | AVDN M | AVDN R | AVSN S | AVSN M | AVSN R | TR Acc | AVH Acc* |", "|---|" + "---|" * 9]
for t, v in res.items():
    L.append(f"| **{t}** | " + " | ".join(f"{v[c]:.2f}" for c in cols) + " |")
for t, v in PAPER.items():
    L.append(f"| {t} | " + " | ".join(f"{v[c]:.2f}" for c in cols) + " |")
S = ["| model | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
for t, v in res.items():
    S.append(f"| {t} | " + " | ".join(f"{v['judge_S_all_categories'][k]['S']:.3f} [{v['judge_S_all_categories'][k]['ci95'][0]:.2f}, {v['judge_S_all_categories'][k]['ci95'][1]:.2f}]" for k in CATS) + " |")
P = []
for name, d in pairs.items():
    P.append(f"| {name} | " + " | ".join(f"{d[k]['mean_diff']:+.3f} [{d[k]['ci95'][0]:+.2f}, {d[k]['ci95'][1]:+.2f}]" for k in CATS) + f" | {d['TR_acc_pts']['mean_diff']:+.1f} [{d['TR_acc_pts']['ci95'][0]:+.1f}, {d['TR_acc_pts']['ci95'][1]:+.1f}] |")
X = []
for t, v in res.items():
    X.append(f"| {t} | {v['TR']:.1f} [{v['TR_ci95'][0]}, {v['TR_ci95'][1]}] | " + " / ".join(f"{v['AVH_by_subtype'][k]:.1f}" for k in ("Action", "Object", "Sound")) +
             f" | {v['AVH_yes_rate']:.1f} % | " + " / ".join(f"{v['median_output_to_gold_word_ratio'][k]:.2f}" for k in ("AVSN", "AVDN", "SSA", "TR")) + " |")
open(f"{out_dir}/tables.md", "w").write("\n".join(["### Main (paper Table 3 columns)", *L, "", "### Judge S, all 3,976 items per category (95% CI)", *S, "",
    "### Paired differences (same items; 95% CI)", "| pair | " + " | ".join(CATS) + " | TR Acc (pts) |", "|---|" + "---|" * (len(CATS) + 1), *P, "",
    "### Closed-ended details", "| model | TR-MCQ Acc [95% CI] | AVH yes/no Acc Action / Object / Sound | AVH 'Yes' rate | median output/gold words AVSN / AVDN / SSA / TR |", "|---|---|---|---|---|", *X]) + "\n")
print(open(f"{out_dir}/tables.md").read())
