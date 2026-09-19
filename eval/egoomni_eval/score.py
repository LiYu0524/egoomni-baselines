#!/usr/bin/env python
"""Aggregate predictions + judgments into accuracy tables.
  python -m egoomni_eval.score --tag salmonn2plus_7b            # gold protocol (headline)
  python -m egoomni_eval.score --tag salmonn2plus_7b --protocol self   # single-turn rows (from gold) + self-history multi-turn rows
"""
import argparse
import json
import os
from collections import defaultdict

from . import EVAL_DIR
from .data import read_jsonl, read_pred_dir
from .judge import pred_hash
from .prompts import parse_mcq


def load_rows(pred_root, tag, protocol):
    """gold: all gold-protocol rows. self: single-turn rows (protocol gold) + multi-turn rows under the self protocol."""
    rows = read_pred_dir(os.path.join(pred_root, tag))
    if protocol == "gold":
        return [r for r in rows if r["protocol"] == "gold"]
    return [r for r in rows if (r["meta"]["source_kind"] != "multi_turn" and r["protocol"] == "gold") or
            (r["meta"]["source_kind"] == "multi_turn" and r["protocol"] == "self")]


def load_judgments(jroot, tag):
    j = {}
    for row in read_jsonl(os.path.join(jroot, f"{tag}.jsonl")):
        if row.get("correct") is not None:
            j[(row["key"], row["pred_hash"])] = row["correct"]
    return j


def decide(row, judgments):
    """→ (correct: bool, status: 'ok'|'infer_error'|'unjudged'|'unparsed_mcq')"""
    if row.get("pred") is None:
        return False, "infer_error"
    if row["fmt"] == "mcq":
        letter = parse_mcq(row["pred"], row["options"])
        if letter is None:
            return False, "unparsed_mcq"
        return letter in [c.upper() for c in row["correct_options"]], "ok"
    c = judgments.get((row["key"], pred_hash(row["pred"])))
    if c is None:
        return False, "unjudged"
    return bool(c), "ok"


def table(rows, keyfn, name):
    g = defaultdict(lambda: [0, 0])
    for r in rows:
        k = keyfn(r)
        if k is None:
            continue
        g[k][0] += 1; g[k][1] += r["_correct"]
    out = {str(k): {"n": n, "acc": round(c / n, 4)} for k, (n, c) in sorted(g.items(), key=lambda kv: str(kv[0]))}
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--protocol", default="gold", choices=["gold", "self"])
    ap.add_argument("--pred_root", default=f"{EVAL_DIR}/preds")
    ap.add_argument("--judg_root", default=f"{EVAL_DIR}/judgments")
    ap.add_argument("--out_root", default=f"{EVAL_DIR}/results")
    a = ap.parse_args()

    rows = load_rows(a.pred_root, a.tag, a.protocol)
    # dedupe by key (later shards/reruns win)
    rows = list({(r["key"], r["protocol"]): r for r in rows}.values())
    J = load_judgments(a.judg_root, a.tag)
    status = defaultdict(int)
    for r in rows:
        r["_correct"], st = decide(r, J)
        status[st] += 1
    n = len(rows)
    multi = [r for r in rows if r["meta"]["source_kind"] == "multi_turn"]
    single = [r for r in rows if r["meta"]["source_kind"] != "multi_turn"]
    # all rounds correct per conversation
    conv = defaultdict(list)
    for r in multi:
        conv[r["item_id"]].append(r["_correct"])
    all_correct = [all(v) and len(v) == max(1, next((r["n_turns"] for r in multi if r["item_id"] == k), 0)) for k, v in conv.items()]

    res = {
        "tag": a.tag, "protocol": a.protocol, "n_rows": n, "status": dict(status),
        "overall_acc": round(sum(r["_correct"] for r in rows) / max(n, 1), 4),
        "single_turn_acc": round(sum(r["_correct"] for r in single) / max(len(single), 1), 4),
        "multi_turn_round_acc": round(sum(r["_correct"] for r in multi) / max(len(multi), 1), 4),
        "multi_turn_all_rounds_correct": round(sum(all_correct) / max(len(all_correct), 1), 4),
        "n_conversations": len(all_correct),
        "mean_latency_s": round(sum(r.get("latency_s") or 0 for r in rows) / max(n, 1), 2),
        "breakdowns": {},
    }
    for name, out in [
        table(rows, lambda r: r["fmt"], "format"),
        table(rows, lambda r: r["meta"]["track"], "track"),
        table(rows, lambda r: r["meta"]["category"], "category"),
        table(rows, lambda r: r["meta"]["subcategory"], "subcategory"),
        table(rows, lambda r: r["meta"]["min_modalities"] or "unspecified", "min_modalities"),
        table(rows, lambda r: "audio+video" if r["has_audio"] else "video-only clip", "clip_audio"),
        table(rows, lambda r: r["modality_used"] or "n/a", "modality_used"),
        table(rows, lambda r: r["lang"], "lang"),
        table(multi, lambda r: f"round{r['turn_idx']}", "multi_turn_round"),
        table(rows, lambda r: ("<30s" if r["duration"] < 30 else "30-120s" if r["duration"] < 120 else "2-10min" if r["duration"] < 600 else ">10min"), "clip_duration"),
    ]:
        res["breakdowns"][name] = out

    od = os.path.join(a.out_root, a.tag)
    os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, f"{a.protocol}.json"), "w"), indent=2, ensure_ascii=False)
    md = [f"# {a.tag} — protocol {a.protocol}", "",
          f"rows: {n} | status: {dict(status)} | mean latency {res['mean_latency_s']} s", "",
          "| metric | value |", "|---|---|",
          f"| overall acc | {res['overall_acc']:.4f} |", f"| single-turn acc (n={len(single)}) | {res['single_turn_acc']:.4f} |",
          f"| multi-turn round acc (n={len(multi)}) | {res['multi_turn_round_acc']:.4f} |",
          f"| multi-turn all-rounds-correct (n={len(all_correct)}) | {res['multi_turn_all_rounds_correct']:.4f} |", ""]
    for name, out in res["breakdowns"].items():
        md += [f"## {name}", "", "| group | n | acc |", "|---|---|---|"] + [f"| {k} | {v['n']} | {v['acc']:.4f} |" for k, v in out.items()] + [""]
    open(os.path.join(od, f"{a.protocol}.md"), "w").write("\n".join(md))
    print("\n".join(md[:12]))
    print(f"→ {od}/{a.protocol}.md")


if __name__ == "__main__":
    main()
