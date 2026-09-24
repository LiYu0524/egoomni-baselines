#!/usr/bin/env python3
"""Markdown tables across the five ego benchmarks (missing = "…").
Table 1 — original protocol (LLaMAFactory predict, EgoAVU training media recipe) for every Qwen2.5-Omni-7B model evaluated on H.
Table 2 — the colleague's EgoToM-kit protocol (native HF + qwen_omni_utils, 2 fps <= 360 frames, 156,800 px, audio-in-video) for
          the colleague's ModelScope models and the base; each cell "kit (Δ vs original protocol)".
Table 3 — same model, kit vs original protocol, paired per question (EgoSchema / EgoCross letter correctness): exact McNemar.
  EgoAVU-Bench: official judge S (SSA, AVSN), METEOR (AVSN), TR / AVH accuracy   egoOmni: gold / self overall (all 5,176 gold rows)
  EgoCross: official closed-set accuracy   EgoSchema: Subset accuracy (lmms-eval parse)   EgoTaskQA: direct / indirect (EM, judge)
usage: summarize_all.py [OUT_MD]"""
import json, math, sys

L = "/mnt/shared-storage-user/ai4good1-share/liyu"
BI, EO = f"{L}/egoavu/bench_infer", f"{L}/egoOmni_baselines/eval"
MODELS = [  # (label, EAB tag, egoOmni tag, tag in the EgoCross / EgoSchema / EgoTaskQA kits)
    ("base (untuned Qwen2.5-Omni-7B)", "base_qwen25omni7b", "qwen25omni7b_base", "base"),
    ("r100k (ours)", "r100k", "egoavu_r100k", "r100k"),
    ("r20k-8gpu (ours)", "r20k8g", "egoavu_r20k8g", "r20k8g"),
    ("r20k-32gpu (ours)", "r20k32g", "egoavu_r20k32g", "r20k32g"),
    ("ckpt_sft (colleague LoRA)", "ckpt_sft", "ckpt_sft", "ckpt_sft"),
    ("ckpt_epoch2 = ckpt_lora_epoch2 (colleague LoRA)", "ckpt_epoch2", "ckpt_epoch2", "ckpt_epoch2"),
    ("ckpt_fft_epoch2 (colleague full FT)", "ckpt_fft_epoch2", "ckpt_fft_epoch2", "ckpt_fft_epoch2"),
]
KIT = [("base (untuned Qwen2.5-Omni-7B)", "base", 0), ("ckpt_epoch2 = ckpt_lora_epoch2 (colleague LoRA)", "ckpt_epoch2", 5),
       ("ckpt_fft_epoch2 (colleague full FT)", "ckpt_fft_epoch2", 6)]
HEAD = ["model", "EAB SSA S", "EAB AVSN S", "EAB AVSN M", "EAB TR", "EAB AVH", "egoOmni gold", "egoOmni self", "EgoCross",
        "EgoSchema", "ETQ direct EM", "ETQ direct judge", "ETQ indirect EM", "ETQ indirect judge"]
ND = [2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def eab_models(*paths):
    out = {}
    for p in paths:
        d = load(p)
        if d:
            out.update(d["models"])
    return out


def values(e, og, os_, x, s, q):
    g = lambda d, *k: None if not d else (d[k[0]] if len(k) == 1 else (d[k[0]] or {}).get(k[1]))
    return [e.get("SSA"), e.get("AVSN_S"), e.get("AVSN_M"), e.get("TR"), e.get("AVH"),
            og and 100 * og["overall_acc"], os_ and 100 * os_["overall_acc"], x and x["official_parse"]["acc"],
            s and s["acc_lmms_eval_parse"], g(q, "direct", "em"), g(q, "direct", "judge"), g(q, "indirect", "em"), g(q, "indirect", "judge")]


def model_values(eab, te, to, tk):
    return values(eab.get(te, {}), load(f"{EO}/results/{to}/gold.json"), load(f"{EO}/results/{to}/self.json"),
                  load(f"{L}/egocross_eval_h/results/{tk}/scores.json"), load(f"{L}/egoschema_eval_h/results/{tk}/scores.json"),
                  load(f"{L}/egotaskqa_eval_h/results/{tk}/scores.json"))


def fmt(v, nd):
    return "…" if v is None else f"{v:.{nd}f}"


def table(rows):
    return "\n".join(["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)] + ["| " + " | ".join(r) + " |" for r in rows])


def mcnemar(a, b):   # a, b: {qid: correct}; exact two-sided
    ks = sorted(set(a) & set(b)); n01 = sum(a[k] and not b[k] for k in ks); n10 = sum(b[k] and not a[k] for k in ks)
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
    return len(ks), n01, n10, p


def correct(bench, tag):
    if bench == "egoschema":
        rows = [json.loads(l) for l in open(f"{L}/egoschema_eval_h/results/{tag}/outputs.jsonl")]
        return {r["question_idx"]: r["pred_lmms"] == r["gold"] for r in rows}
    rows = [json.loads(l) for l in open(f"{L}/egocross_eval_h/results/{tag}/outputs.jsonl")]
    return {r["question_id"]: (r["pred_official"] or "").strip().upper()[:1] == r["gold"] for r in rows}


eab_old = eab_models(f"{BI}/results_h/summary.json", f"{BI}/results_h_fft/summary.json")
old = [model_values(eab_old, *m[1:]) for m in MODELS]
md = ["**Table 1 — original protocol** (LLaMAFactory, EgoAVU media recipe: 2 fps ≤ 64 frames, 200,704 px)", "",
      table([[m[0]] + [fmt(v, nd) for v, nd in zip(vals, ND)] for m, vals in zip(MODELS, old)]), "",
      "**Table 2 — colleague's EgoToM-kit protocol** (native HF + qwen_omni_utils, 2 fps ≤ 360 frames, 156,800 px, audio-in-video; "
      "cell = kit (Δ vs Table 1))", ""]
rows = []
for label, m, i in KIT:
    eab_kit = eab_models(f"{BI}/results_h_kit_{m}/summary.json")
    kit = values(eab_kit.get(f"kit_{m}", {}), load(f"{EO}/results/kit_{m}/gold.json"), load(f"{EO}/results/kit_{m}/self.json"),
                 load(f"{L}/egocross_eval_h/results/kit_{m}/scores.json"), load(f"{L}/egoschema_eval_h/results/kit_{m}/scores.json"),
                 load(f"{L}/egotaskqa_eval_h/results/kit_{m}/scores.json"))
    rows.append([label] + [fmt(k, nd) if k is None or o is None else f"{k:.{nd}f} ({k - o:+.{nd}f})" for k, o, nd in zip(kit, old[i], ND)])
md += [table(rows), "", "**Table 3 — kit vs original protocol, same model, paired per question** (exact McNemar; letter correctness)", "",
       "| model | benchmark | n | original only correct | kit only correct | p |", "|---|---|---|---|---|---|"]
for label, m, i in KIT:
    for bench in ("egoschema", "egocross"):
        try:
            n, a, b, p = mcnemar(correct(bench, MODELS[i][3]), correct(bench, f"kit_{m}"))
            md.append(f"| {label} | {bench} | {n} | {a} | {b} | {p:.3g} |")
        except (OSError, KeyError):
            md.append(f"| {label} | {bench} | … | … | … | … |")
md = "\n".join(md) + "\n"
print(md)
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(md)
