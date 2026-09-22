#!/usr/bin/env python3
"""EgoAVU official LLM-as-judge (evaluation/llm_as_judge.py @ 15bd5bb), reproduced faithfully with vLLM for throughput.
Identical to upstream: judge model Qwen3-235B-A22B-Instruct-2507; JUDGE_PROMPT read verbatim from the upstream file (only the two
literal JSON braces escaped, without which upstream str.format raises KeyError); single user message through the tokenizer's
chat template with add_generation_prompt=True; greedy decoding (upstream passes temperature=0.0), max_new_tokens 512;
skip_special_tokens decode; parse = json.loads(output) -> int(parsed["rating"]), unparseable items skipped; per-(file, category)
mean rounded to 3 decimals; CSV columns file_name, category, score.
Additions (do not change the official scores): per-item raw judge outputs, per-category counts of scored / skipped items.
usage: judge_official.py INPUT_DIR OUT_DIR JUDGE_MODEL_DIR [TP]"""
import ast, csv, json, os, sys, time
from collections import defaultdict

UPSTREAM = "/shared/egoavu/repo/evaluation/llm_as_judge.py"

def load_items(json_path):                                   # upstream load_items
    with open(json_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    elif isinstance(data, list):
        return data
    raise ValueError(f"Invalid JSON format in {json_path}")


def parse_judge_output(output):                              # upstream parse_judge_output
    try:
        parsed = json.loads(output)
        return int(parsed["rating"])
    except Exception:
        return None


def main():
    input_dir, out_dir, model_dir = sys.argv[1:4]
    tp = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    os.makedirs(out_dir, exist_ok=True)

    tree = ast.parse(open(UPSTREAM).read())
    P0 = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "JUDGE_PROMPT")
    P = P0.replace('{\n  "rating"', '{{\n  "rating"').replace('"<brief explanation>"\n}', '"<brief explanation>"\n}}')
    assert P.replace("{{", "{").replace("}}", "}") == P0 and P.count("{{") == 1 and P.count("}}") == 1

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(model_dir)
    jobs = []
    for file_name in sorted(os.listdir(input_dir)):
        if not file_name.endswith(".json"):
            continue
        for item in load_items(os.path.join(input_dir, file_name)):
            category = item["category"]
            grounding = item.get("answer", item.get("ground_truth"))
            prediction = item.get("output", item.get("prediction"))
            question = item.get("question", item.get("question"))
            prompt = P.format(question=question, grounding=grounding, prediction=prediction)
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            jobs.append(dict(file_name=file_name, bench_idx=item.get("bench_idx"), category=category, text=text))
    print(f"[judge] {len(jobs)} prompts from {len({j['file_name'] for j in jobs})} files", flush=True)
    print("[judge] example prompt:\n" + jobs[0]["text"][:1500], flush=True)

    llm = LLM(model=model_dir, tensor_parallel_size=tp, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92)
    sp = SamplingParams(temperature=0.0, max_tokens=512, skip_special_tokens=True)
    t0 = time.time()
    outs = llm.generate([j["text"] for j in jobs], sp, use_tqdm=True)
    print(f"[judge] generated {len(outs)} in {time.time()-t0:.0f}s", flush=True)

    scores, counts = defaultdict(list), defaultdict(lambda: [0, 0])
    with open(os.path.join(out_dir, "judge_items.jsonl"), "w") as f:
        for j, o in zip(jobs, outs):
            raw = o.outputs[0].text
            r = parse_judge_output(raw)
            key = (j["file_name"], j["category"])
            counts[key][0] += 1
            if r is not None:
                scores[key].append(r); counts[key][1] += 1
            f.write(json.dumps({"file_name": j["file_name"], "bench_idx": j["bench_idx"], "category": j["category"], "rating": r,
                                "finish_reason": o.outputs[0].finish_reason, "judge_output": raw}, ensure_ascii=True) + "\n")
    rows = [{"file_name": fn, "category": c, "score": round(sum(v) / len(v), 3)} for (fn, c), v in scores.items()]
    with open(os.path.join(out_dir, "judge_scores.csv"), "w", newline="") as f:          # upstream output format
        w = csv.DictWriter(f, fieldnames=["file_name", "category", "score"]); w.writeheader(); w.writerows(rows)
    json.dump({f"{fn}|{c}": {"items": n, "scored": s, "skipped_unparseable": n - s} for (fn, c), (n, s) in counts.items()},
              open(os.path.join(out_dir, "judge_counts.json"), "w"), indent=1)
    print(open(os.path.join(out_dir, "judge_scores.csv")).read())


if __name__ == "__main__":
    main()
