#!/usr/bin/env python3
"""egoOmni instances for the EgoToM-kit-protocol re-test: eval/kit_<model> for base, ckpt_epoch2 (= groo ckpt_lora_epoch2) and
ckpt_fft_epoch2, cloned from eval/base (same datasets, same gold/self protocol, same join checks); only the predict step differs:
tools/h/nat_predict_lf.py (native Qwen2.5-Omni, the kit's settings, 256 new tokens as for every egoOmni model, <=360 frames)
instead of LLaMAFactory. Also writes the EgoAVU-Bench shard files egoavu_bench_eval_s{k}of30 (bench_idx % 30 == k).
usage: make_kit_instances.py"""
import json, os

L = "/mnt/shared-storage-user/ai4good1-share/liyu"
E = f"{L}/egoOmni_baselines/eval"
P = "/ai4good1-shared/liyu"
MODELS = {"base": (f"{P}/egoavu/models/Qwen2.5-Omni-7B-thinker-train", "none"),
          "ckpt_epoch2": (f"{P}/egoavu/models/Qwen2.5-Omni-7B-thinker-train", f"{P}/egoavu/bench_infer/adapters/ckpt_epoch2"),
          "ckpt_fft_epoch2": (f"{P}/egoavu/models/groo_ckpt_fft_epoch2", "none")}
JOB = """#!/usr/bin/env bash
# rjob worker (started by tools/h/kit_job_h.sh, which restores the env and mounts the media): {model} on egoOmni shard k of K with
# the colleague's EgoToM-kit protocol (tools/h/nat_predict_lf.py in place of LLaMAFactory predict) - gold protocol, then
# self-protocol rounds 2..4 of the same items, exactly as eval/base/job.sh.   usage: job.sh K k
set -uo pipefail
K=$1; k=$2
R={R}; M={M}; AD={AD}
mkdir -p $R/logs $R/runs; exec > >(tee -a $R/logs/job_s${{k}}of${{K}}_$(hostname).log) 2>&1
echo "[{inst}] $(date -Is) host=$(hostname) shard=$k/$K model=$M adapter=$AD"
PY=/shared/egoavu/env/bin/python
predict() {{   # $1 = dataset name
  local ds=$1 n t; n=$(wc -l < $R/data/$ds.jsonl)
  [ "$n" -eq 0 ] && {{ echo "[{inst}] $ds empty"; return 0; }}
  if [ -f $R/runs/$ds/generated_predictions.jsonl ] && [ "$(wc -l < $R/runs/$ds/generated_predictions.jsonl)" -eq "$n" ]; then echo "[{inst}] $ds already predicted"
  else t=$(date +%s); $PY /ai4good1-shared/liyu/tools/h/nat_predict_lf.py $R/data/$ds.jsonl $R/runs/$ds $M $AD 256 360 >> $R/logs/nat_$ds.log 2>&1 \\
         || {{ tail -3 $R/logs/nat_$ds.log; return 1; }}
       echo "[{inst}] $(date -Is) $ds: $n rows in $(( $(date +%s) - t )) s"; fi
  (cd $R && $PY collect.py $ds $K $k)
}}
cd $R
predict gold_s${{k}}of${{K}} || {{ echo "[{inst}] gold predict FAILED"; exit 1; }}
for rn in 2 3 4; do
  (cd $R && $PY build_data.py self $K $k $rn) && predict self_r${{rn}}_s${{k}}of${{K}} || {{ echo "[{inst}] self round $rn FAILED"; exit 1; }}
done
(cd $R && $PY collect.py none $K $k --final)
echo "[{inst}] $(date -Is) DONE shard $k/$K"
"""


def clone(src, dst, subs):
    s = open(src).read()
    for a, b in subs:
        assert s.count(a) == 1, (src, a)
        s = s.replace(a, b)
    open(dst + ".tmp", "w").write(s); os.chmod(dst + ".tmp", 0o755); os.replace(dst + ".tmp", dst)


for m, (M, AD) in MODELS.items():
    inst = f"kit_{m}"; R = f"{P}/egoOmni_baselines/eval/{inst}"; d = f"{E}/{inst}"
    os.makedirs(d, exist_ok=True)
    clone(f"{E}/base/build_data.py", f"{d}/build_data.py",
          [(f'R = "{P}/egoOmni_baselines/eval/base"', f'R = "{R}"'),
           ("for the EgoAVU r100k LoRA on the egoOmni bench", f"for {m} on the egoOmni bench (EgoToM-kit protocol re-test)")])
    clone(f"{E}/base/collect.py", f"{d}/collect.py",
          [(f'TAG, MODEL = "qwen25omni7b_base", "{P}/egoavu/models/Qwen2.5-Omni-7B-thinker-train"',
            f'TAG, MODEL = "{inst}", "{M}" + ("" if "{AD}" == "none" else " + {AD}") + " [EgoToM-kit protocol]"'),
           ("eval/preds/qwen25omni7b_base/", f"eval/preds/{inst}/")])
    open(f"{d}/job.sh.tmp", "w").write(JOB.format(model=m, inst=inst, R=R, M=M, AD=AD))
    os.chmod(f"{d}/job.sh.tmp", 0o755); os.replace(f"{d}/job.sh.tmp", f"{d}/job.sh")
    if not os.path.lexists(f"{d}/audio16k"):
        os.symlink("../r100k/audio16k", f"{d}/audio16k")
    print("instance", inst)

BI = f"{L}/egoavu/bench_infer"
rows = [json.loads(l) for l in open(f"{BI}/data/egoavu_bench_eval.jsonl")]
for k in range(30):
    p = f"{BI}/data/egoavu_bench_eval_s{k}of30.jsonl"
    with open(p + ".tmp", "w") as f:
        for r in rows:
            if r["bench_idx"] % 30 == k:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)
print("EgoAVU-Bench shards s{k}of30 written:", sum(1 for k in range(30) for _ in open(f"{BI}/data/egoavu_bench_eval_s{k}of30.jsonl")), "rows")
