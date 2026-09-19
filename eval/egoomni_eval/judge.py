#!/usr/bin/env python
"""LLM judge for open-ended answers. Backends: vllm (local Qwen3-32B, TP=2) | openai (any OpenAI-compatible endpoint).
Judgments are cached by (key, protocol, hash(pred)); MCQ rows are scored rule-based in score.py and skipped here.

  python -m egoomni_eval.judge --tag salmonn2plus_7b --backend vllm --judge_model /ai4good1-shared/models/Qwen3-32B --tp 2
  JUDGE_BASE_URL=... JUDGE_API_KEY=... python -m egoomni_eval.judge --tag ... --backend openai --judge_model gpt-5.4
"""
import argparse
import hashlib
import json
import os
import re
import time

from . import EVAL_DIR
from .data import read_jsonl, read_pred_dir

JUDGE_VERSION = "v1"
SYSTEM = ("You are a strict but fair grader for an egocentric video question-answering benchmark. "
          "Questions, reference answers and model answers may be in English or Chinese.")
USER_TPL = """{context}[Question]
{question}

[Reference answer]
{gold}

[Model answer]
{pred}

Decide whether the model answer is CORRECT with respect to the reference answer.
- Correct: it conveys the same key facts as the reference (the object/person/action/location/order/sound/speech that the question asks about). Different wording, more detail, or a different language are fine as long as nothing contradicts the reference.
- Incorrect: it misses or contradicts the main point, is vague or hedged, answers a different question, refuses, or says it cannot tell.
Reply with JSON only: {{"correct": true or false, "reason": "<one short sentence>"}}"""
CTX_TPL = "[Earlier turns of the same conversation, for context]\n{turns}\n\n"


def pred_hash(pred: str) -> str:
    return hashlib.sha1((pred or "").encode()).hexdigest()[:12]


def build_prompt(row: dict) -> list[dict]:
    ctx = ""
    if row["turn_idx"] > 1:
        msgs = row["messages"][:-1]
        lines = []
        for i in range(0, len(msgs), 2):
            q = msgs[i]["content"].split("\n")[0]
            a = msgs[i + 1]["content"] if i + 1 < len(msgs) else ""
            lines.append(f"Q{i//2+1}: {q}\nA{i//2+1}: {a}")
        ctx = CTX_TPL.format(turns="\n".join(lines))
    user = USER_TPL.format(context=ctx, question=row["question"].strip(), gold=(row["gold"] or "").strip(), pred=(row["pred"] or "").strip())
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_JSON = re.compile(r"\{.*?\}", re.S)


def parse_verdict(text: str):
    for m in _JSON.finditer(text or ""):
        try:
            d = json.loads(m.group(0))
            if "correct" in d:
                c = d["correct"]
                if isinstance(c, str):
                    c = c.strip().lower() in ("true", "yes", "1", "correct")
                return bool(c), str(d.get("reason", ""))[:300]
        except json.JSONDecodeError:
            continue
    t = (text or "").strip().lower()
    if re.match(r"^(correct|true|yes)\b", t):
        return True, text[:300]
    if re.match(r"^(incorrect|false|no)\b", t):
        return False, text[:300]
    return None, (text or "")[:300]


class VllmBackend:
    def __init__(self, model, tp, max_model_len=6144, gpu_util=0.9):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_model_len, gpu_memory_utilization=gpu_util, dtype="bfloat16")
        self.sp = SamplingParams(temperature=0, max_tokens=256)

    def run(self, prompts: list[list[dict]]) -> list[str]:
        outs = self.llm.chat(prompts, self.sp, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
        return [o.outputs[0].text for o in outs]


class OpenAIBackend:
    def __init__(self, model, concurrency=16):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.environ["JUDGE_BASE_URL"], api_key=os.environ["JUDGE_API_KEY"])
        self.model, self.conc = model, concurrency

    def _one(self, msgs):
        for attempt in range(4):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0, max_tokens=256)
                return r.choices[0].message.content
            except Exception as e:  # noqa: BLE001
                time.sleep(2 ** attempt)
                err = e
        return f"__error__ {err}"

    def run(self, prompts):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.conc) as ex:
            return list(ex.map(self._one, prompts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "openai"])
    ap.add_argument("--judge_model", default="/ai4good1-shared/models/Qwen3-32B")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu_util", type=float, default=0.85, help="vLLM gpu_memory_utilization (fraction of each GPU); lower it to co-run with inference")
    ap.add_argument("--pred_root", default=f"{EVAL_DIR}/preds")
    ap.add_argument("--out_root", default=f"{EVAL_DIR}/judgments")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    rows = [r for r in read_pred_dir(os.path.join(a.pred_root, a.tag)) if r["fmt"] != "mcq" and r.get("pred") is not None]
    op = os.path.join(a.out_root, f"{a.tag}.jsonl")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    cache = {(j["key"], j["pred_hash"]) for j in read_jsonl(op) if j.get("correct") is not None}
    todo, seen = [], set()
    for r in rows:                                   # identical (key, pred) across protocols is judged once
        h = (r["key"], pred_hash(r["pred"]))
        if h not in cache and h not in seen:
            todo.append(r); seen.add(h)
    if a.limit:
        todo = todo[:a.limit]
    print(f"[judge] {a.tag}: open rows={len(rows)} unique={len(seen)+len(cache & {(r['key'], pred_hash(r['pred'])) for r in rows})} cached={len(rows)-len(todo)} todo={len(todo)}", flush=True)
    if not todo:
        return
    be = VllmBackend(a.judge_model, a.tp, gpu_util=a.gpu_util) if a.backend == "vllm" else OpenAIBackend(a.judge_model)
    judge_name = os.path.basename(a.judge_model.rstrip("/"))
    n_bad = 0
    with open(op, "a") as f:
        for s in range(0, len(todo), a.batch):
            chunk = todo[s:s + a.batch]
            texts = be.run([build_prompt(r) for r in chunk])
            # one retry for unparseable verdicts
            redo = [i for i, t in enumerate(texts) if parse_verdict(t)[0] is None]
            if redo:
                t2 = be.run([build_prompt(chunk[i]) + [{"role": "assistant", "content": texts[i]}, {"role": "user", "content": "Reply with the JSON object only."}] for i in redo])
                for i, t in zip(redo, t2):
                    texts[i] = t
            for r, t in zip(chunk, texts):
                c, reason = parse_verdict(t)
                n_bad += c is None
                f.write(json.dumps({"key": r["key"], "pred_hash": pred_hash(r["pred"]), "correct": c, "reason": reason,
                                    "judge": judge_name, "judge_version": JUDGE_VERSION, "raw": None if c is not None else t[:500]}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[judge] {min(s+a.batch, len(todo))}/{len(todo)} unparseable so far={n_bad}", flush=True)


if __name__ == "__main__":
    main()
