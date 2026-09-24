#!/usr/bin/env python3
"""EgoTaskQA scoring for one model. Joins the 8 shard predictions back to the questions (by order; verified: echoed label ==
gold answer) and reports two accuracies per split and per EgoTaskQA breakdown (type / category / semantic / structural):
  EM    - normalized exact match (lowercase, punctuation and articles removed); yes/no questions use the first word only
  judge - Qwen3-32B (vLLM, thinking off, greedy) decides whether a non-yes/no prediction means the same as the reference;
          yes/no questions keep the EM decision.
usage: score_judge.py TAG JUDGE_MODEL_DIR"""
import collections, json, os, re, sys
R = "/ai4good1-shared/liyu/egotaskqa_eval_h"; tag, judge_dir = sys.argv[1], sys.argv[2]
meta = json.load(open(f"{R}/data/meta.json"))
ART = {"a", "an", "the"}
norm = lambda s: " ".join(w for w in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if w not in ART)
rows = []
for ds, items in meta.items():
    gen = [json.loads(l) for l in open(f"{R}/runs/{tag}/{ds}/generated_predictions.jsonl")]
    assert len(gen) == len(items), f"{ds}: {len(gen)} vs {len(items)}"
    for it, g in zip(items, gen):
        assert g["label"].strip() == it["answer"].strip(), f"join mismatch {it['question_id']}"
        pred, gold = norm(g["predict"]), norm(it["answer"])
        yn = gold in ("yes", "no")
        em = (pred.split()[:1] == [gold]) if yn else (pred == gold)
        rows.append({**it, "split": ds.split("_")[1], "output": g["predict"], "yesno": yn, "em": em})
todo = [r for r in rows if not r["yesno"] and not r["em"]]
from vllm import LLM, SamplingParams
llm = LLM(model=judge_dir, tensor_parallel_size=1, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
tok = llm.get_tokenizer()
def prompt(r):
    msg = ("You grade short answers to questions about an egocentric video. Decide whether the predicted answer means the "
           "same thing as the reference answer for this question (synonyms and paraphrases are fine; a different object, action, "
           "state or attribute is wrong).\n\n"
           f"Question: {r['question']}\nReference answer: {r['answer']}\nPredicted answer: {r['output'].strip()}\n\n"
           "Reply with exactly one word: yes or no.")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
outs = llm.generate([prompt(r) for r in todo], SamplingParams(temperature=0, max_tokens=4))
unp = 0
for r, o in zip(todo, outs):
    t = o.outputs[0].text.strip().lower()
    r["judge_raw"] = t; r["judge"] = t.startswith("yes"); unp += not (t.startswith("yes") or t.startswith("no"))
for r in rows:
    r.setdefault("judge", r["em"])
def acc(sub, key):
    return round(100 * sum(r[key] for r in sub) / len(sub), 2) if sub else None
res = {"tag": tag, "judge_unparsed": unp, "judged_items": len(todo)}
for split in ("direct", "indirect"):
    S = [r for r in rows if r["split"] == split]
    d = {"n": len(S), "em": acc(S, "em"), "judge": acc(S, "judge"), "yesno_acc": acc([r for r in S if r["yesno"]], "em"),
         "open_em": acc([r for r in S if not r["yesno"]], "em"), "open_judge": acc([r for r in S if not r["yesno"]], "judge")}
    for dim in ("type", "category", "semantic", "structural"):
        groups = collections.defaultdict(list)
        for r in S:
            for v in (r[dim] if isinstance(r[dim], list) else [r[dim]]):
                groups[v].append(r)
        d[dim] = {k: {"n": len(v), "em": acc(v, "em"), "judge": acc(v, "judge")} for k, v in sorted(groups.items())}
    res[split] = d
out = f"{R}/results/{tag}"; os.makedirs(out, exist_ok=True)
with open(f"{out}/outputs.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(res, open(f"{out}/scores.json", "w"), indent=2)
print(json.dumps({k: (v if not isinstance(v, dict) else {x: v[x] for x in ("n", "em", "judge", "yesno_acc", "open_em", "open_judge")}) for k, v in res.items()}))
