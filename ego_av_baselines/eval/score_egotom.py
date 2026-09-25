#!/usr/bin/env python3
"""Score EgoToM multiple-choice predictions.

usage: score_egotom.py MODEL [MODEL ...]      (reads preds/<model>/egotom/shard_*.jsonl, writes results/<model>/egotom.json)
Ground truth: egotom/egotom_<q>_shuffled.csv — gt_<q> is the text of the correct option among <q>_choice_{a,b,c,...}.
The official prompt asks for "Answer <question>: <option>) <answer>"; the chosen letter is taken from that pattern, else from
the first "x)" / "(x)" / "option x" in the reply, else from a unique verbatim option text in the reply; otherwise the item
counts as wrong (unparsed rate is reported). Accuracy per context condition x question type, micro-averaged over all
questions, for the full EgoToM release and the paper subset (egotom_paper/).
"""
import csv, glob, json, os, re, sys
from collections import defaultdict

E = os.path.dirname(os.path.abspath(__file__))
REPO = "/root/egobench_probe/egotom_repo"


def gold():
    g = {}
    for q in ("goal", "belief", "actions"):
        for r in csv.DictReader(open(f"{REPO}/egotom/egotom_{q}_shuffled.csv")):
            opts = {k.rsplit("_", 1)[1]: v for k, v in r.items() if k.startswith(f"{q}_choice_") and v}
            ans = [l for l, v in opts.items() if v.strip() == r[f"gt_{q}"].strip()]
            assert len(ans) == 1, (q, r["cuid"])
            g[(q, r["cuid"])] = (ans[0], opts)
    paper = set()
    for q in ("goal", "belief", "actions"):
        for r in csv.DictReader(open(f"{REPO}/egotom_paper/egotom_{q}_shuffled.csv")):
            paper.add((q, r["cuid"]))
    return g, paper


def parse(pred, opts):
    t = pred.strip()
    letters = "".join(sorted(opts))
    lead = r"Answer\s*(?:Q(?:uestion)?\s*)?\d*\s*:\s*"            # "Answer 1:", "Answer Q1:", "Answer Question 1:"
    m = re.search(rf"{lead}\(?([{letters}])\)", t, re.I) or re.search(rf"{lead}([{letters}])\b", t, re.I)
    if not m:
        m = re.search(rf"(?:^|[\s(\[])([{letters}])\)", t) or re.search(rf"\(([{letters}])\)", t) or \
            re.search(rf"\b(?:option|answer is)\s*\(?([{letters}])\b", t, re.I) or re.fullmatch(rf"\s*([{letters}])\s*[.)]?\s*", t, re.I)
    if m:
        return m.group(1).lower()
    hits = [l for l, v in opts.items() if v.strip().rstrip(".").lower() in t.lower()]
    return hits[0] if len(hits) == 1 else None


def main():
    g, paper = gold()
    for model in sys.argv[1:]:
        preds = {}
        for f in glob.glob(f"{E}/preds/{model}/egotom/shard_*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if "pred" in r:
                    preds[r["id"]] = r["pred"]
        acc = defaultdict(lambda: [0, 0, 0])            # key -> [correct, total, unparsed]
        missing = 0
        for (q, cuid), (ans, opts) in g.items():
            for cond in ("fullcontext", "last30sec", "last5sec"):
                pid = f"{cond}|{q}|{cuid}"
                if pid not in preds:
                    missing += 1
                    continue
                p = parse(preds[pid], opts)
                ok = p == ans
                for split in (["all", "paper"] if (q, cuid) in paper else ["all"]):
                    for key in ((split, cond, q), (split, cond, "overall")):
                        acc[key][0] += ok
                        acc[key][1] += 1
                        acc[key][2] += p is None
        res = {f"{s}|{c}|{q}": {"acc": round(100 * v[0] / v[1], 2), "n": v[1], "unparsed": v[2]} for (s, c, q), v in sorted(acc.items())}
        res["_missing_predictions"] = missing
        os.makedirs(f"{E}/results/{model}", exist_ok=True)
        json.dump(res, open(f"{E}/results/{model}/egotom.json", "w"), indent=1)
        print(model, "missing", missing)
        for s in ("paper", "all"):
            print("  ", s, {c: res.get(f"{s}|{c}|overall", {}).get("acc") for c in ("fullcontext", "last30sec", "last5sec")})


if __name__ == "__main__":
    main()
