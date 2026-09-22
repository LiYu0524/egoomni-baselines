#!/usr/bin/env python3
"""EgoAVU-Bench closed-ended accuracy (paper: "for close-ended QAs ... we follow Yue et al., 2024 [MMMU] and use regex-based string
matching ... to extract key phrases such as option IDs (A, B, C, D), binary indicators (yes/no), and conclusion phrases").
The paper's module is not released; this is an explicit re-implementation, reported with its extraction-rule counts.
Closed-ended items are identified from the GOLD format of the released CSV:
  TR-MCQ: gold "The correct option is X, which is ..." (500 items, 4 options "A: ...")
  AVH yes/no: gold starting with "Yes"/"No" (304 items; action / object / sound)
MCQ extraction order (the models mostly answer by reproducing an option's text): (1) response == one option's text (normalized),
(2) explicit phrase ("correct option is X", "answer is X", "option X"), (3) a letter followed by ':' '.' ')' or a lone letter
(NOT the article in "A person ..."), (4) exactly one option's text contained in the response, (5) token-F1 best option >= 0.6
with a >= 0.1 margin over the runner-up, (6) otherwise unparsed = wrong. MMMU's parse_multi_choice_response is also run verbatim
(random fallback, seed 42) for reference only: it needs > 5 words for text matching and reads the article "A" as option A.
Yes/no extraction: (1) first word yes/no, (2) first standalone yes/no word, (3) negation cue (not, n't, no <noun>, never, none,
nothing, absent) -> no, (4) otherwise an affirmative statement -> yes (counted separately as 'heuristic_affirmative').
usage: closed_ended_acc.py [--dump DIR] PRED_JSON [PRED_JSON ...]  -> prints JSON summary per file (+ per-item extractions)"""
import csv, json, random, re, sys
from collections import Counter, defaultdict

CSV = "/ai4good1-shared/liyu/egoavu/data/EgoAVU_data/egoavu_bench_combined.csv"
rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8")))
GOLD_MCQ = re.compile(r"^\s*The correct option is ([A-D])\b")
OPT_LINE = re.compile(r"^([A-D]):\s*(.+?)\s*$", re.M)
YN = re.compile(r"^\s*(yes|no)\b", re.I)

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()

def _f1(a, b):
    a, b = norm(a).split(), norm(b).split()
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b); common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(a), common / len(b)
    return 2 * pr * rc / (pr + rc)

def mcq_extract(resp, opts):
    r = resp.strip(); nr = norm(r)
    hits = [k for k, v in opts.items() if norm(v) == nr]                       # (1) the option text itself
    if len(hits) == 1:
        return hits[0], "exact_option_text"
    m = re.search(r"(?i:correct (?:option|answer|choice)|answer|option|choice)\s*(?i:is|would be|:)?\s*\(?\b([A-D])\b\)?", r)  # letter is case-sensitive: "answer a call" is not option A
    if m and m.group(1).upper() in opts:                                       # (2) explicit phrase
        return m.group(1).upper(), "phrase"
    m = re.match(r"^\W*\(?([A-D])\)?\s*(?:[\.\):]|$)", r)                     # (3) letter + ':' '.' ')' or a lone letter
    if m and m.group(1) in opts:
        return m.group(1), "letter"
    hits = [k for k, v in opts.items() if norm(v) and norm(v) in nr]           # (4) exactly one option text contained
    if len(hits) == 1:
        return hits[0], "contains_option_text"
    sc = sorted(((_f1(r, v), k) for k, v in opts.items()), reverse=True)       # (5) clear fuzzy best match
    if sc[0][0] >= 0.6 and sc[0][0] - sc[1][0] >= 0.1:
        return sc[0][1], "fuzzy_option_text"
    return None, "unparsed"


def mmmu_parse(response, all_choices, index2ans, rng):          # MMMU eval_utils.parse_multi_choice_response, verbatim logic
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    index_ans, ans_with_brack, candidates = True, False, []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice); ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index); index_ans = False
    if len(candidates) == 0:
        return rng.choice(all_choices), True
    if len(candidates) > 1:
        if index_ans:
            idx = [response.rfind(f"({c})") if ans_with_brack else response.rfind(f" {c} ") for c in candidates]
        else:
            idx = [response.lower().rfind(index2ans[c].lower()) for c in candidates]
        return candidates[max(range(len(idx)), key=lambda i: idx[i])], False
    return candidates[0], False

NEG = re.compile(r"\b(not|never|none|nothing|nobody|no one|absent|neither|nor|without)\b|n't\b|\bno\s+\w+", re.I)
def yn_extract(resp):
    r = resp.strip()
    m = YN.match(r.lstrip("\"'*([ "))
    if m:
        return m.group(1).lower(), "first_word"
    m = re.search(r"\b(yes|no)\b", r, re.I)
    if m:
        return m.group(1).lower(), "yes_no_word"
    if NEG.search(r):
        return "no", "negation_cue"
    return "yes", "heuristic_affirmative"

def score(path, dump=None):
    items = json.load(open(path)); by = {it["bench_idx"]: it for it in items}
    per_item = []
    rng = random.Random(42)
    out = {"file": path.split("/")[-1]}
    # TR MCQ
    res, rules, mmmu_ok, mmmu_rand = [], Counter(), 0, 0
    for i, r in enumerate(rows):
        g = GOLD_MCQ.match(r["answer"])
        if r["category"] != "Temporal Reasoning" or not g:
            continue
        opts = dict(OPT_LINE.findall(r["question"])); assert set(opts) == set("ABCD"), i
        pred, rule = mcq_extract(by[i]["output"], opts); rules[rule] += 1
        res.append(pred == g.group(1))
        per_item.append({"bench_idx": i, "task": "TR_mcq", "gold": g.group(1), "pred": pred, "rule": rule, "correct": pred == g.group(1)})
        mp, rnd = mmmu_parse(by[i]["output"], list("ABCD"), opts, rng); mmmu_ok += (mp == g.group(1)); mmmu_rand += rnd
    out["TR_mcq"] = {"n": len(res), "acc": round(100 * sum(res) / len(res), 2), "extraction": dict(rules),
                     "mmmu_parser_acc": round(100 * mmmu_ok / len(res), 2), "mmmu_random_fallbacks": mmmu_rand}
    # AVH yes/no
    per, rules, preds = defaultdict(list), Counter(), Counter()
    for i, r in enumerate(rows):
        if not r["category"].startswith("Audio Visual Hallucination"):
            continue
        g = YN.match(r["answer"])
        if not g:
            continue
        pred, rule = yn_extract(by[i]["output"]); rules[rule] += 1; preds[pred] += 1
        per[r["category"]].append(pred == g.group(1).lower())
        per_item.append({"bench_idx": i, "task": "AVH_yesno", "subtype": r["category"].split("(")[1].rstrip(")"),
                         "gold": g.group(1).lower(), "pred": pred, "rule": rule, "correct": pred == g.group(1).lower()})
    allv = [x for v in per.values() for x in v]
    out["AVH_yesno"] = {"n": len(allv), "acc": round(100 * sum(allv) / len(allv), 2),
                        "by_subtype": {k.split("(")[1].rstrip(")"): {"n": len(v), "acc": round(100 * sum(v) / len(v), 2)} for k, v in sorted(per.items())},
                        "extraction": dict(rules), "predicted": dict(preds)}
    sub = out["AVH_yesno"]["by_subtype"]
    out["AVH_yesno"]["macro_acc"] = round(sum(v["acc"] for v in sub.values()) / len(sub), 2)
    if dump:
        with open(dump, "w") as f:
            for it in per_item:
                f.write(json.dumps(it) + "\n")
    return out

if __name__ == "__main__":
    args = sys.argv[1:]
    dump_dir = None
    if args and args[0] == "--dump":
        dump_dir, args = args[1], args[2:]
    for p in args:
        d = f"{dump_dir}/{p.split('/')[-1].replace('.json', '')}_closed_items.jsonl" if dump_dir else None
        print(json.dumps(score(p, d)))
