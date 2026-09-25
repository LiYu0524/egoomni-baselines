#!/usr/bin/env python3
"""LLM-judge EgoSound / EgoTempo open-ended predictions with each benchmark's official judge prompt (run on openvla).

usage: judge_openended.py BENCH MODEL [--judge NAME] [--workers N]
  egosound: EgoSound qa_eval_gpt.py — system + user messages, official judge model gpt-5, max_completion_tokens 800,
            reply "{'binary_pred': ..., 'score': ...}"; accuracy = binary_pred == 'correct' over judged items.
  egotempo: EgoTempo gemini_eval.ipynb create_prompt() sent as one user message (official judge gemini-1.5-flash is not
            offered by the gateway; default here gemini-2.5-flash); reply "{'pred': ..., 'score': ..., 'reason': ...}";
            accuracy = pred == 'correct'.
Credentials: /root/.gpt_judge.env (OPENAI_API_KEY, OPENAI_BASE_URL; chmod 600, never copied elsewhere).
Judgments are cached in judgments/<bench>/<model>__<judge>.jsonl (resumable); scores -> results/<model>/<bench>.json.
"""
import argparse, ast, glob, json, os, re, threading, time, unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests

E = os.path.dirname(os.path.abspath(__file__))
P = "/root/egobench_probe"
for l in open("/root/.gpt_judge.env"):
    k, _, v = l.strip().partition("=")
    os.environ.setdefault(k, v)
URL, KEY = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions", os.environ["OPENAI_API_KEY"]
for k in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
    os.environ.pop(k, None)                                    # the gateway is reached directly from openvla
lock = threading.Lock()


def egosound_msgs(q, a, pred):                                 # qa_eval_gpt.build_judge_messages, verbatim
    sys_msg = (
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs "
        "for video-based question-answer pairs. Your task is to compare the predicted answer with the "
        "correct answer and determine if the predicted answer is correct or not.\n"
        "------\n"
        "## INSTRUCTIONS:\n"
        "- Focus on correctness and factual accuracy.\n"
        "- The predicted answer must align with the video content implied by the question/answer.\n"
        "- The predicted answer and the correct answer may differ in language; translate mentally and compare semantics.\n"
        "- Synonyms/paraphrases are valid.\n"
        "- Output ONLY a Python dictionary string with keys 'binary_pred' and 'score'.\n"
        "- 'binary_pred' must be 'correct' or 'incorrect'.\n"
        "- 'score' must be an integer from 0 (fully wrong) to 5 (fully correct).\n"
        "- No extra text, no markdown.\n"
        "Example: {'binary_pred': 'correct', 'score': 4}"
    )
    user_msg = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {q}\n"
        f"Correct Answer: {a}\n"
        f"Predicted Answer: {pred}\n\n"
        "Return ONLY a Python dictionary string like: {'binary_pred': 'correct', 'score': 4}."
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def egotempo_msgs(q, a, pred):                                  # gemini_eval.ipynb create_prompt, verbatim
    return [{"role": "user", "content": f"""role: "system",
content: "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs.
Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:
-----##INSTRUCTIONS:
- Focus on the correctness and accuracy of the predicted answer with the ground-truth.
- Consider uncertain predictions, such as 'it is impossible to answer the question from the video', as incorrect, unless the ground truth answer also says that."
role: "user",
content: "Please evaluate the following video-based question-answer pair:
Question: {q}
Ground truth correct Answer: {a}
Predicted Answer: {pred}
Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness.
Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is a string of 'correct' or 'incorrect',
value of 'score' is in INTEGER, not STRING and value of 'reason' should provide the reason behind the decision."
"""}]


def call(judge, msgs, max_tokens):
    body = {"model": judge, "messages": msgs}
    body["max_completion_tokens" if judge.startswith(("gpt-5", "o")) else "max_tokens"] = max_tokens
    last = None
    for attempt in range(8):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=300)
            if r.status_code == 200:
                c = r.json()["choices"][0]["message"].get("content") or ""
                if c.strip():
                    return c
                last = "empty response"
            else:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last = str(e)[:200]
        time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(last)


def parse_dict(text):
    """Official parsing (first {...} -> ast.literal_eval); if the reply is cut inside the free-text 'reason', fall back to
    the verdict/score fields, which come first in both official formats."""
    t = unicodedata.normalize("NFKC", text).strip()
    m = re.search(r"\{.*?\}", t, re.S)
    try:
        return ast.literal_eval(m.group(0) if m else t)
    except Exception:
        v = re.search(r"['\"](binary_pred|pred)['\"]\s*:\s*['\"](correct|incorrect)['\"]", t)
        sc = re.search(r"['\"]score['\"]\s*:\s*['\"]?(\d)", t)
        if not v:
            raise
        return {v.group(1): v.group(2), "score": int(sc.group(1)) if sc else 0, "_partial": True}


def gold(bench):
    if bench == "egosound":
        g = {}
        for f in ("ego4d", "egoblind"):
            for a in json.load(open(f"{P}/egosound_{f}.json")):
                g[a["question_id"]] = {"q": a["question"], "a": a["answer"], "subset": f, "type": a["question_type"]}
        return g
    return {a["question_id"]: {"q": a["question"], "a": a["answer"], "type": a["question_type"]}
            for a in json.load(open(f"{P}/egotempo_openQA.json"))["annotations"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["egosound", "egotempo"])
    ap.add_argument("model")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    judge = a.judge or {"egosound": "gpt-5", "egotempo": "gemini-2.5-flash"}[a.bench]
    g = gold(a.bench)
    preds = {}
    for f in glob.glob(f"{E}/preds/{a.model}/{a.bench}/shard_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if "pred" in r:
                preds[r["id"]] = r["pred"]
    os.makedirs(f"{E}/judgments/{a.bench}", exist_ok=True)
    cache = f"{E}/judgments/{a.bench}/{a.model}__{judge}.jsonl"
    done = {}
    if os.path.exists(cache):
        for l in open(cache):
            r = json.loads(l)
            if "verdict" in r:
                done[r["id"]] = r
    todo = [i for i in g if i in preds and i not in done]
    print(f"{a.bench}/{a.model} judge={judge}: gold {len(g)}, preds {len(preds)}, judged {len(done)}, to judge {len(todo)}", flush=True)
    mk = egosound_msgs if a.bench == "egosound" else egotempo_msgs

    def one(i):
        rec = {"id": i}
        try:
            raw = call(judge, mk(g[i]["q"], g[i]["a"], preds[i]), 800 if a.bench == "egosound" else 4096)
            d = parse_dict(raw)
            v = d.get("binary_pred", d.get("pred", ""))
            rec.update(verdict=str(v).strip().lower(), score=int(d.get("score", 0)), raw=raw[:500])
        except Exception as e:
            rec["error"] = str(e)[:300]
        with lock, open(cache, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
    with ThreadPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(one, todo)):
            if "verdict" in r:
                done[r["id"]] = r
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(todo)}", flush=True)
    agg = defaultdict(lambda: [0, 0, 0.0])
    for i, r in done.items():
        if i not in g:
            continue
        keys = ["overall", f"type:{g[i]['type']}"] + ([f"subset:{g[i]['subset']}"] if "subset" in g[i] else [])
        for k in keys:
            agg[k][0] += r["verdict"] == "correct"
            agg[k][1] += 1
            agg[k][2] += r["score"]
    res = {k: {"acc": round(100 * v[0] / v[1], 2), "mean_score": round(v[2] / v[1], 3), "n": v[1]} for k, v in sorted(agg.items())}
    res["_meta"] = {"judge": judge, "gold": len(g), "with_pred": len(preds), "judged": len(done),
                    "missing_pred": len(set(g) - set(preds)), "judge_errors": len(todo) - sum(1 for i in todo if i in done)}
    os.makedirs(f"{E}/results/{a.model}", exist_ok=True)
    json.dump(res, open(f"{E}/results/{a.model}/{a.bench}__{judge}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("overall", "_meta") if k in res}))


if __name__ == "__main__":
    main()
