#!/usr/bin/env python3
"""EgoCross: join predictions to questions (by order; verified: the echoed label equals the gold letter), parse the letter
exactly like EgoCross/eval_os_closedset.py (json.loads, else first {...} block -> "prediction"), and score with the official
Codabench rules (codabench_v2/scoring_program/scoring.py: normalize_ans, letter-or-text match, per domain).
Also reports a lenient parse (bare option letter) and the 916 questions whose frames are all present.  usage: collect.py TAG"""
import collections, json, os, re, sys
R = "/ai4good1-shared/liyu/egocross_eval_h"; E = "/mnt/shared-storage-user/liyu/egocross"
sys.path.insert(0, E + "/codabench_v2/scoring_program")
import scoring  # noqa: E402

tag = sys.argv[1]
meta = json.load(open(f"{R}/data/meta.json"))


def official(resp):
    try:
        return str(json.loads(resp).get("prediction", "") or ""), "primary"
    except Exception:
        try:
            m = re.search(r'\{[\s\S]*?\}', resp)
            if m:
                return str(json.loads(m.group(0)).get("prediction", "") or ""), "secondary"
        except Exception:
            pass
    return "", "unparsed"


def lenient(resp):
    p, _ = official(resp)
    if p:
        return p
    for pat in (r'"?prediction"?\s*[:=]\s*"?\(?([A-D])\b', r'^\s*\(?([A-D])[\):.\s]', r'\b(?:answer|option)\s*(?:is\s*)?[:(]?\s*([A-D])\b'):
        m = re.search(pat, resp, re.I if "answer" in pat else 0)
        if m:
            return m.group(1).upper()
    return ""


rows = []
for ds, items in meta.items():
    gen = [json.loads(l) for l in open(f"{R}/runs/{tag}/{ds}/generated_predictions.jsonl")]
    assert len(gen) == len(items), f"{ds}: {len(gen)} predictions vs {len(items)} questions"
    for it, g in zip(items, gen):
        assert g["label"].strip() == it["gold"], f"join mismatch at {it['question_id']}"
        p, how = official(g["predict"])
        rows.append({**it, "output": g["predict"], "pred_official": p, "parse": how, "pred_lenient": lenient(g["predict"])})
gt = scoring.load_answers(f"{E}/codabench_v2/reference_data/answers.json")
cat = {r["question_id"]: r["category"] for r in rows}


def score(pm, keep=None):
    tot = cor = 0; dom = collections.defaultdict(lambda: [0, 0]); cats = collections.defaultdict(lambda: [0, 0])
    for qid, gold in gt.items():
        if keep is not None and qid not in keep:
            continue
        p = scoring.normalize_ans(pm.get(qid))
        ok = p is not None and (p == gold["letter"] or p == gold["text"])
        tot += 1; cor += ok; dom[gold["domain"]][0] += 1; dom[gold["domain"]][1] += ok
        cats[cat.get(qid)][0] += 1; cats[cat.get(qid)][1] += ok
    pct = lambda c, t: round(100 * c / t, 2)
    return {"acc": pct(cor, tot), "n": tot, "by_domain": {d: pct(c, t) for d, (t, c) in sorted(dom.items())},
            "by_category": {k: pct(c, t) for k, (t, c) in sorted(cats.items())}}


off = {r["question_id"]: r["pred_official"] for r in rows}
complete = {r["question_id"] for r in rows if not r["incomplete"]}
res = {"tag": tag, "official_parse": score(off), "official_parse_complete_frames": score(off, complete),
       "lenient_parse": score({r["question_id"]: r["pred_lenient"] for r in rows}),
       "parse_status": dict(collections.Counter(r["parse"] for r in rows))}
out = f"{R}/results/{tag}"; os.makedirs(out, exist_ok=True)
json.dump(off, open(f"{out}/predictions.json", "w"), indent=1)          # Codabench submission format {question_id: letter}
with open(f"{out}/outputs.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(res, open(f"{out}/scores.json", "w"), indent=2)
print(json.dumps(res))
