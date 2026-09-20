#!/usr/bin/env python
"""Sum API spend from prediction rows; write STOP_API when the cap is hit. Run alongside API workers.
  python cost_monitor.py --tag gemini_3_8_flash --cap 100 --total_requests 6315 [--interval 60]"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from egoomni_eval import EVAL_DIR  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--cap", type=float, required=True)
ap.add_argument("--total_requests", type=int, default=6315); ap.add_argument("--interval", type=int, default=60); ap.add_argument("--once", action="store_true")
a = ap.parse_args()
stop = os.path.join(EVAL_DIR, "STOP_API")
t0, first = time.time(), None
while True:
    rows = [json.loads(l) for f in glob.glob(f"{EVAL_DIR}/preds/{a.tag}/shard*.jsonl") for l in open(f) if l.strip()]
    gen = [r for r in rows if r.get("cost_usd") is not None]
    spend = sum(r["cost_usd"] for r in gen); n = len(gen); err = sum(1 for r in rows if r.get("pred") is None)
    tok = {k: sum(r["usage"][k] for r in gen) for k in ("prompt", "video", "audio", "text", "output", "thoughts")} if gen else {}
    proj = spend / n * a.total_requests if n else 0
    print(f"[{time.strftime('%H:%M:%S')}] requests={n}/{a.total_requests} errors={err} spend=${spend:.2f} projected_total=${proj:.2f} "
          f"avg/req=${spend/max(n,1):.4f} tokens={tok}", flush=True)
    if spend >= a.cap and not os.path.exists(stop):
        open(stop, "w").write(f"cap {a.cap} reached: spend {spend:.2f} at {time.strftime('%F %T')}\n"); print("!!! CAP REACHED — STOP_API written", flush=True)
    if a.once or (n >= a.total_requests):
        break
    time.sleep(a.interval)
