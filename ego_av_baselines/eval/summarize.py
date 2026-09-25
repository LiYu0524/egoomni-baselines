#!/usr/bin/env python3
"""Assemble the final EgoSound / EgoToM / EgoTempo table (openvla) -> results/SUMMARY.md + results/summary.json.

Published cells (from the user's table) are kept; our runs fill the blanks. Our re-runs of published cells are listed
separately as calibration. EgoToM is reported for every context condition on both the paper subset and the full release.
"""
import json, os

E = os.path.dirname(os.path.abspath(__file__))
R = f"{E}/results"
MODELS = [("egogpt", "EgoGPT-7B"), ("salmonn2p", "video-SALMONN2+-7B"), ("minicpmo", "MiniCPM-o 2.6-8B"),
          ("qwen2vl", "Qwen2-VL-7B"), ("llavaov", "LLaVA-OneVision-Qwen2-7B")]
PUBLISHED = {("egogpt", "egosound"): 34.30, ("salmonn2p", "egosound"): 36.00, ("minicpmo", "egosound"): 40.40,
             ("qwen2vl", "egotempo"): 26.10, ("llavaov", "egotempo"): 23.30}
TEMPO_MAIN, TEMPO_ALT = "gemini-2.5-flash-lite", "gemini-2.5-flash"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    out, lines = {}, []
    for m, name in MODELS:
        es = load(f"{R}/{m}/egosound__gpt-5.json")
        tm = load(f"{R}/{m}/egotempo__{TEMPO_MAIN}.json")
        ta = load(f"{R}/{m}/egotempo__{TEMPO_ALT}.json")
        et = load(f"{R}/{m}/egotom.json")
        out[m] = {"egosound": es and es["overall"], "egotempo_main": tm and tm["overall"], "egotempo_alt": ta and ta["overall"],
                  "egotom": et}
    lines.append("## Main table (published cells kept; ours fill the blanks)\n")
    lines.append("| Model | EgoSound | EgoToM (paper set, last30sec) | EgoTempo |")
    lines.append("|---|---|---|---|")
    for m, name in MODELS:
        cells = []
        for b in ("egosound", "egotom", "egotempo"):
            if (m, b) in PUBLISHED:
                cells.append(f"{PUBLISHED[(m, b)]:.2f} (pub.)")
            elif b == "egosound":
                cells.append(f"**{out[m]['egosound']['acc']:.2f}**" if out[m]["egosound"] else "…")
            elif b == "egotempo":
                cells.append(f"**{out[m]['egotempo_main']['acc']:.2f}**" if out[m]["egotempo_main"] else "…")
            else:
                v = (out[m]["egotom"] or {}).get("paper|last30sec|overall")
                cells.append(f"**{v['acc']:.2f}**" if v else "…")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("\n## EgoToM, all conditions (accuracy %, n = questions answered)\n")
    lines.append("| Model | split | fullcontext | last30sec | last5sec | unparsed | missing preds |")
    lines.append("|---|---|---|---|---|---|---|")
    for m, name in MODELS:
        et = out[m]["egotom"]
        if not et:
            continue
        for s in ("paper", "all"):
            c = [et.get(f"{s}|{k}|overall") for k in ("fullcontext", "last30sec", "last5sec")]
            unp = sum(x["unparsed"] for x in c if x)
            lines.append(f"| {name} | {s} | " + " | ".join(f"{x['acc']:.2f} (n={x['n']})" if x else "…" for x in c)
                         + f" | {unp} | {et.get('_missing_predictions')} |")
    lines.append(f"\n## Calibration (our pipeline on published cells)\n")
    lines.append("| Model | Bench | Published | Ours | Judge |")
    lines.append("|---|---|---|---|---|")
    for (m, b), pub in PUBLISHED.items():
        if b == "egotempo" and out[m]["egotempo_main"]:
            lines.append(f"| {dict(MODELS)[m]} | EgoTempo | {pub:.2f} | {out[m]['egotempo_main']['acc']:.2f} "
                         f"(alt {out[m]['egotempo_alt']['acc']:.2f}) | {TEMPO_MAIN} (alt {TEMPO_ALT}) |")
        if b == "egosound" and out[m]["egosound"]:
            lines.append(f"| {dict(MODELS)[m]} | EgoSound | {pub:.2f} | {out[m]['egosound']['acc']:.2f} | gpt-5 |")
    open(f"{R}/SUMMARY.md", "w").write("\n".join(lines) + "\n")
    json.dump(out, open(f"{R}/summary.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
