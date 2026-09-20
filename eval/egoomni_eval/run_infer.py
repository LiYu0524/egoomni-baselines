#!/usr/bin/env python
"""Inference worker: answers every turn of its items under the gold and self protocols, decoding each clip once.
Resumable (per (key, protocol)), errors recorded per row, media prefetched on threads so the GPU stays busy.

  # 7B: 8 independent workers (one GPU each)
  CUDA_VISIBLE_DEVICES=3 python -m egoomni_eval.run_infer --adapter salmonn2plus --model_dir models/video-SALMONN2_plus_7B_full \
      --tag salmonn2plus_7b --shard 3 --nshards 8
  # 72B: ZeRO-3 data-parallel, shard/nshards from torchrun RANK/WORLD_SIZE
  torchrun --nproc_per_node 8 -m egoomni_eval.run_infer --adapter salmonn2plus --model_dir models/video-SALMONN2_plus_72B_full --tag salmonn2plus_72b --zero3
Protocols: gold = turn k conditioned on dataset answers for turns <k; self = conditioned on this model's own answers
(self round 1 is identical to gold round 1 and is copied, not regenerated).
"""
import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from . import EVAL_DIR, ROOT
from .data import build_item_plans, load_clips_meta, load_items, load_subset_ids, read_pred_dir, request_for
from .models import get_adapter
from .models.base import StopWorker


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=["videollama2", "salmonn2plus", "minicpmo", "gemini"])
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tag", required=True, help="output name, e.g. salmonn2plus_7b")
    ap.add_argument("--protocols", default="gold,self")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--zero3", action="store_true")
    ap.add_argument("--subset", default=None, help="json with item_ids (dev subset)")
    ap.add_argument("--limit", type=int, default=None, help="first N items of this shard")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out_root", default=f"{EVAL_DIR}/preds")
    ap.add_argument("--out_name", default=None, help="override output subdir (default = tag)")
    ap.add_argument("--retry_errors", action="store_true")
    ap.add_argument("--prefetch", type=int, default=3, help="clips decoded ahead on background threads")
    ap.add_argument("--cpu_threads", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=768)
    ap.add_argument("--max_pixels", type=int, default=61250)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--no_gpu_preprocess", action="store_true", help="salmonn: resize/normalize on CPU (repo default, ~50x slower)")
    ap.add_argument("--max_units", type=int, default=128, help="minicpmo: max 1-second units per clip (longer clips are uniformly subsampled)")
    return ap.parse_args()


def assign_shards(plans, nshards, protocols):
    """Greedy balance by generation count (multi-turn items cost more), deterministic."""
    plans = sorted(plans, key=lambda p: (-p.n_generations(protocols), p.item_id))
    loads, buckets = [0] * nshards, [[] for _ in range(nshards)]
    for p in plans:
        i = loads.index(min(loads))
        buckets[i].append(p); loads[i] += p.n_generations(protocols)
    return [sorted(b, key=lambda p: p.item_id) for b in buckets]


def main():
    a = parse()
    protocols = [p for p in a.protocols.split(",") if p]
    if a.adapter != "gemini" and not os.path.isabs(a.model_dir):      # API adapters take a model name, not a path
        a.model_dir = os.path.join(ROOT, a.model_dir)
    local_rank = 0
    if a.zero3:
        a.shard, a.nshards, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
        import torch
        torch.cuda.set_device(local_rank)
    tagp = f"[{a.tag} shard {a.shard}/{a.nshards}]"
    log = lambda *x: print(f"[{time.strftime('%H:%M:%S')}]{tagp}", *x, flush=True)

    plans = build_item_plans(load_items(), load_clips_meta(), load_subset_ids(a.subset))
    mine = assign_shards(plans, a.nshards, protocols)[a.shard]
    if a.limit:
        mine = mine[:a.limit]
    op = os.path.join(a.out_root, a.out_name or a.tag, f"shard{a.shard}.jsonl")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    prev = {(r["key"], r["protocol"]): r for r in read_pred_dir(os.path.dirname(op))}   # all shards: safe to change --nshards on resume
    done = {k for k, r in prev.items() if r.get("pred") is not None or not a.retry_errors}
    tk = lambda p, k: f"{p.item_id}#r{k}"

    def plan_turns(p):
        """Yield (proto, k, action) for every turn of item p: action = 'skip' | 'copy' (self r1 from gold r1) | 'gen'."""
        for proto in protocols:
            if proto == "self" and not p.multi:
                continue
            for k in range(1, p.n_turns + 1):
                key = (tk(p, k), proto)
                if key in done:
                    yield proto, k, "skip"
                elif proto == "self" and k == 1 and ("gold" in protocols or (prev.get((key[0], "gold")) or {}).get("pred") is not None):
                    yield proto, k, "copy"
                else:
                    yield proto, k, "gen"

    todo = [p for p in mine if any(act != "skip" for _, _, act in plan_turns(p))]
    gens_of = lambda p: sum(1 for _, _, act in plan_turns(p) if act == "gen")
    n_gen = sum(gens_of(p) for p in todo)
    log(f"items total={len(plans)} mine={len(mine)} todo={len(todo)} generations={n_gen}")

    # ZeRO-3 lockstep: every parameter all-gather is a collective, so all ranks must run the same modules in the same
    # order. Clips without audio skip the audio encoder → process audio and no-audio items in two phases, each padded
    # with dummy generations so every rank makes the same number of generate() calls per phase.
    phases = [("av", [p for p in todo if p.has_audio]), ("v", [p for p in todo if not p.has_audio])] if a.zero3 else [("all", todo)]
    n_pad = {}
    if a.zero3:
        import torch.distributed as dist
        import deepspeed
        deepspeed.init_distributed()
        for name, items in phases:
            n = sum(gens_of(p) for p in items)
            t = torch.tensor([n], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX)
            n_pad[name] = int(t.item()) - n
            log(f"zero3 lockstep phase {name}: mine={n} max across ranks={int(t.item())} padding={n_pad[name]}")
        if sum(n_pad.values()) == 0 and not todo:
            return
    elif not todo:
        log("nothing to do"); return

    Adapter = get_adapter(a.adapter)
    gpus = list(range(len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))))
    adapter = Adapter(a.model_dir, gpus, zero3=a.zero3, max_frames=a.max_frames, max_pixels=a.max_pixels, interval=a.interval,
                      gpu_preprocess=not a.no_gpu_preprocess, cpu_threads=a.cpu_threads, max_units=a.max_units)
    t0 = time.time(); adapter.load(); log(f"model loaded in {time.time()-t0:.0f}s")

    pool = ThreadPoolExecutor(max_workers=max(1, a.prefetch))
    futures = {}
    def submit(i):
        if 0 <= i < len(todo) and i not in futures:
            futures[i] = pool.submit(adapter.prepare_media, todo[i].clip, todo[i].has_audio)

    state = {"ok": 0, "err": 0, "done": 0, "last": {}}     # last[phase] = (media, messages) for lockstep dummies
    t_start = time.time()

    def dummy_gen(phase):
        if phase not in state["last"]:
            want = phase == "av"
            fb = next((p for p in plans if p.has_audio == want), None)
            if fb is None:
                return
            state["last"][phase] = (adapter.prepare_media(fb.clip, fb.has_audio), request_for(fb, 1, "gold", []).messages)
        try:
            adapter.generate(state["last"][phase][0], state["last"][phase][1], max_new_tokens=4)
        except Exception:
            pass

    with open(op, "a") as f:
        def emit(req, res, err, t1, tprep):
            row = req.to_row()
            row.update(model_tag=a.tag, model_dir=a.model_dir, pred=None, error=None, modality_used=None, n_input_tokens=None,
                       latency_s=round(time.time() - t1, 2), prep_wait_s=tprep, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
            if err is None:
                row.update(res); state["ok"] += 1
            else:
                row["error"] = err; state["err"] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            state["done"] += 1
            if state["done"] % 10 == 0 or state["done"] == n_gen:
                el = time.time() - t_start
                log(f"{state['done']}/{n_gen} ok={state['ok']} err={state['err']} {el/state['done']:.1f}s/gen eta={(n_gen-state['done'])*el/state['done']/60:.0f}min")

        todo = [p for _, items in phases for p in items]     # phase order (audio items first under zero3)
        for i in range(a.prefetch):
            submit(i)
        pos = 0
        for name, items in phases:
            for p in items:
                i = pos; pos += 1
                submit(i + a.prefetch)
                tp = time.time()
                try:
                    media, merr = futures.pop(i).result(), None
                except Exception as e:  # noqa: BLE001
                    media, merr = None, f"prepare_media {type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
                    log(f"MEDIA ERROR {p.item_id}: {str(e)[:160]}")
                tprep = round(time.time() - tp, 2)      # wait for media (≈0 when prefetch keeps up)
                answers = {}                             # (proto, k) -> pred
                for proto, k, act in plan_turns(p):
                    key = (tk(p, k), proto)
                    if act == "skip":
                        r = prev.get(key)
                        if r and r.get("pred") is not None:
                            answers[(proto, k)] = r["pred"]
                        continue
                    if act == "copy":
                        src = prev.get((key[0], "gold")) or {}
                        g = answers.get(("gold", 1), src.get("pred"))
                        answers[("self", 1)] = g
                        if g is not None:
                            emit(request_for(p, 1, "self", []), {"pred": g, "modality_used": src.get("modality_used") or (media or {}).get("used"),
                                                                 "n_input_tokens": src.get("n_input_tokens")}, None, time.time(), 0.0)
                        continue
                    hist = [p.turns[j]["gold"] for j in range(k - 1)] if proto == "gold" else [answers.get(("self", j + 1)) for j in range(k - 1)]
                    req = request_for(p, k, proto, hist)
                    t1 = time.time()
                    if merr is not None:
                        emit(req, None, merr, t1, tprep)
                        if a.zero3:
                            dummy_gen(name)
                        continue
                    try:
                        res = adapter.generate(media, req.messages, max_new_tokens=a.max_new_tokens)
                        answers[(proto, k)] = res["pred"]
                        state["last"][name] = (media, req.messages)
                        emit(req, res, None, t1, tprep)
                    except StopWorker as e:
                        log(f"STOP: {e} — exiting cleanly (rows so far are saved; rerun resumes)")
                        pool.shutdown(wait=False, cancel_futures=True)
                        return
                    except Exception as e:  # noqa: BLE001
                        emit(req, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", t1, tprep)
                        log(f"ERROR {req.key}/{proto}: {type(e).__name__}: {str(e)[:160]}")
                        try:
                            import torch; torch.cuda.empty_cache()
                        except Exception:
                            pass
                    tprep = 0.0
                media = None
            for _ in range(n_pad.get(name, 0)):      # lockstep padding for this phase (never saved)
                dummy_gen(name)
    pool.shutdown(wait=False)
    log(f"done ok={state['ok']} err={state['err']} elapsed={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
