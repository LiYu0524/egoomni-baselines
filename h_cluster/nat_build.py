#!/usr/bin/env python3
"""Request files for nat_infer.py (the colleague's EgoToM-kit protocol) built from the exact rows the LLaMAFactory runs used,
so question text, options, instructions and media are identical and only the inference protocol changes:
  system  = "You are a helpful assistant." (Qwen2.5-Omni's chat-template default = what every LLaMAFactory run received;
            these benchmarks ship no system prompt of their own, unlike EgoToM)
  user    = the benchmark prompt without LLaMAFactory's leading <video> placeholder (the video goes first in the turn, as before)
  video   = the whole proxy file (the rows' #t=0,<dur> fragment is the full file) at 2 fps, or EgoCross's frame list at its
            protocol rate (0.5 fps; 1.0 for the CholecTrack20 png set)
  label   = the gold answer (kept for the join back)
  sample_id = <dataset>:<row index>
usage: nat_build.py egoschema|egotaskqa|egocross   -> /ai4good1-shared/liyu/natkit/requests/<bench>.jsonl"""
import json, os, sys

L = "/ai4good1-shared/liyu"
SYSTEM = "You are a helpful assistant."
BENCH = {
    "egoschema": (f"{L}/egoschema_eval_h/data", [("egoschema_subset_s0of2", None), ("egoschema_subset_s1of2", None)]),
    "egotaskqa": (f"{L}/egotaskqa_eval_h/data", [(f"etq_{s}_s{k}of4", None) for s in ("direct", "indirect") for k in range(4)]),
    "egocross": (f"{L}/egocross_eval_h/data", [("egocross_fps05_s0of2", 0.5), ("egocross_fps05_s1of2", 0.5), ("egocross_fps10", 1.0)]),
}
bench = sys.argv[1]
ddir, sets = BENCH[bench]
out = []
for ds, frame_fps in sets:
    for i, line in enumerate(open(f"{ddir}/{ds}.jsonl")):
        r = json.loads(line)
        (u, a), v = r["messages"], r["videos"][0]
        assert u["role"] == "user" and a["role"] == "assistant" and u["content"].startswith("<video>") and u["content"].count("<video>") == 1
        assert not r.get("audios"), "these benchmarks have no separate audio"
        req = {"sample_id": f"{ds}:{i}", "dataset": ds, "row": i, "system": SYSTEM, "user": u["content"][len("<video>"):], "label": a["content"]}
        if isinstance(v, list):
            assert frame_fps is not None
            req.update(video=v, sample_fps=frame_fps)
        else:
            path, frag = v.split("#t=") if "#t=" in v else (v, None)
            if frag is not None:
                assert float(frag.split(",")[0]) == 0.0, v          # full-file windows only
            req.update(video=path, fps=2.0)
        out.append(req)
os.makedirs(f"{L}/natkit/requests", exist_ok=True)
p = f"{L}/natkit/requests/{bench}.jsonl"
with open(p + ".tmp", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
os.replace(p + ".tmp", p)
print(f"{bench}: {len(out)} requests from {len(sets)} datasets -> {p}")
