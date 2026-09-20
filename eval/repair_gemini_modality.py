#!/usr/bin/env python
"""One-off repair of the first gemini_3_8_flash run: set modality_used from what was sent (the transcode preserves the
source audio stream) instead of from AUDIO token itemization, which the API omits in ~34% of responses (folding those
tokens into VIDEO — verified: same total tokens, same tokens/second, and 40/40 sampled transcodes have audio streams).
Adds audio_itemized. Backs up each shard to .bak first. Idempotent."""
import glob, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egoomni_eval import EVAL_DIR  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "gemini_3_8_flash"
changed = total = 0
for f in sorted(glob.glob(f"{EVAL_DIR}/preds/{tag}/shard*.jsonl")):
    if not os.path.exists(f + ".bak"):
        shutil.copy2(f, f + ".bak")
    out = []
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line); total += 1
        u = r.get("usage")
        if u is not None:
            r.setdefault("audio_itemized", bool(u.get("audio")))
            want = "av" if r["has_audio"] else "v"
            if r.get("modality_used") != want:
                r["modality_used"] = want; changed += 1
        out.append(json.dumps(r, ensure_ascii=False))
    open(f, "w").write("\n".join(out) + "\n")
print(f"{tag}: rows={total} modality_used corrected={changed}")
