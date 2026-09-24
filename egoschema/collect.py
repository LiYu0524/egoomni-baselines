#!/usr/bin/env python3
"""EgoSchema Subset scoring. Join predictions to questions (by order; verified: echoed label == gold letter).
Primary parse = lmms-eval's extract_characters_regex (strip answer prefixes, first A-E). Because every EgoSchema option
starts with "C ..." (the camera wearer), a model that copies option text is read as "C" by that rule, so a lenient parse is
also reported: a leading "X." / "(X)" letter, else the option whose text the answer contains/matches best.  usage: collect.py TAG"""
import collections, difflib, json, os, re, sys
R = "/ai4good1-shared/liyu/egoschema_eval_h"; tag = sys.argv[1]
meta = json.load(open(f"{R}/data/meta.json"))
PREFIXES = ["The best answer is", "The correct answer is", "The answer is", "The answer", "The best option is",
            "The correct option is", "Best answer:", "Best option:", "Answer:", "Option:"]


def lmms(s):
    s = s.strip()
    for p in PREFIXES:
        s = s.replace(p, "")
    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    m = re.search(r"[ABCDE]", s)
    return m[0] if m else ""


def norm(t):
    return re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()


def lenient(s, options):
    m = re.match(r"^\s*\(?([A-E])(?:[\.\):]|\s*$)", s)
    if m:
        return m.group(1)
    texts = [re.sub(r"^[A-E]\.\s*", "", o) for o in options]
    ns = " ".join(norm(s))
    hit = [i for i, t in enumerate(texts) if " ".join(norm(t)) and " ".join(norm(t)) in ns]
    if hit:
        return "ABCDE"[max(hit, key=lambda i: len(texts[i]))]
    r = [difflib.SequenceMatcher(None, ns, " ".join(norm(t))).ratio() for t in texts]
    return "ABCDE"[r.index(max(r))] if max(r) >= 0.5 else lmms(s)


rows = []
for ds, items in meta.items():
    gen = [json.loads(l) for l in open(f"{R}/runs/{tag}/{ds}/generated_predictions.jsonl")]
    assert len(gen) == len(items), f"{ds}: {len(gen)} vs {len(items)}"
    for it, g in zip(items, gen):
        assert g["label"].strip() == it["gold"], f"join mismatch {it['question_idx']}"
        rows.append({**it, "output": g["predict"], "pred_lmms": lmms(g["predict"]), "pred_lenient": lenient(g["predict"], it["options"])})
acc = lambda k, sub=None: round(100 * sum(r[k] == r["gold"] for r in rows if sub is None or r["has_audio"] == sub) /
                                max(1, sum(1 for r in rows if sub is None or r["has_audio"] == sub)), 2)
res = {"tag": tag, "n": len(rows), "acc_lmms_eval_parse": acc("pred_lmms"), "acc_lenient_parse": acc("pred_lenient"),
       "acc_lenient_with_audio": acc("pred_lenient", True), "acc_lenient_no_audio": acc("pred_lenient", False),
       "n_with_audio": sum(r["has_audio"] for r in rows), "pred_letter_dist_lmms": dict(collections.Counter(r["pred_lmms"] for r in rows)),
       "gold_dist": dict(collections.Counter(r["gold"] for r in rows))}
out = f"{R}/results/{tag}"; os.makedirs(out, exist_ok=True)
with open(f"{out}/outputs.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(res, open(f"{out}/scores.json", "w"), indent=2); print(json.dumps(res))
