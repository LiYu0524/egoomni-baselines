#!/bin/bash
# One-screen status of the EgoSound / EgoToM / EgoTempo baseline evals (run on openvla).
LY=/mnt/shared-storage-user/ai4good1-share/liyu; E=$LY/egobench_eval; D=$LY/tools/egoclips/logs
echo "== $(date '+%F %T')"
echo "clips: egotempo $(grep -c '"ok"' $D/egotempo.jsonl)/367 | egotom stage1 $(grep -c staged $D/egotom-stage1.jsonl)/1305 | egotom final $(grep -c '"ok"' $D/egotom.jsonl)/2355"
source /etc/profile.d/ssh-init.sh >/dev/null 2>&1
echo "jobs: $(brainctl get rjobs -n ailab-safevlagent 2>/dev/null | awk 'NR>1 && $1 ~ /^(es|et|et2|tp|smoke)/ {print $2}' | sort | uniq -c | tr '\n' ' ')"
python3 - <<PY
import json, glob, os
E = "$E"
tot = {"egotom": 3117, "egotempo": 500, "egosound": 7315}
for m in ("qwen2vl", "llavaov", "minicpmo", "egogpt", "salmonn2p"):
    out = []
    for b in ("egosound", "egotom", "egotempo"):
        ok, err = set(), set()
        for f in glob.glob(f"{E}/preds/{m}/{b}/shard_*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                (ok if "pred" in r else err).add(r["id"])
        out.append(f"{b} {len(ok)}/{tot[b]}" + (f" (err {len(err - ok)})" if err - ok else ""))
    print(f"  {m:10s} " + " | ".join(out))
PY
