#!/usr/bin/env python
"""Stratified dev subset: proportional over (source_kind, category, min_modalities, fmt) with ≥1 per stratum,
all-MCQ oversampled, plus the longest clips. → subsets/dev300.json"""
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egoomni_eval import EVAL_DIR, QA_PATH  # noqa: E402
from egoomni_eval.data import item_id  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
random.seed(0)
items = json.load(open(QA_PATH))
strata = defaultdict(list)
for it in items:
    strata[(it["source_kind"], it.get("category"), it.get("minimum_modalities"), it.get("original_question_format"))].append(it)
chosen = {}
for k, v in strata.items():                       # ≥1 per stratum
    chosen[item_id(v[0])] = v[0]
mcq = [it for it in items if it.get("original_question_format") == "mcq"]
for it in random.sample(mcq, 20):                 # enough MCQ to exercise the parser
    chosen[item_id(it)] = it
for it in sorted(items, key=lambda x: -x["clip_duration_sec"])[:4]:   # longest clips
    chosen[item_id(it)] = it
pool = [it for it in items if item_id(it) not in chosen]
random.shuffle(pool)
for it in pool:                                   # fill proportionally (random ≈ proportional)
    if len(chosen) >= N:
        break
    chosen[item_id(it)] = it
ids = sorted(chosen)
out = {"name": f"dev{N}", "n_items": len(ids), "n_turns": sum(len(it["qa"]) for it in chosen.values()), "item_ids": ids}
os.makedirs(f"{EVAL_DIR}/subsets", exist_ok=True)
json.dump(out, open(f"{EVAL_DIR}/subsets/dev{N}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "item_ids"}))
