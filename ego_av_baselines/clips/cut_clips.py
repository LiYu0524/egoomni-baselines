#!/usr/bin/env python3
"""Cut EgoTempo / EgoToM clips from Ego4D sources and store them in s3://safevlagent/liyu/benchmarks/.

usage: cut_clips.py MODE [--workers N]
  egotempo       (openvla) exact [start, end] re-encode from Ego4D video_540ss via the kubebrain proxy
                 -> liyu/benchmarks/egotempo/clips/<official clip name>
  egotom-stage1  (openvla) official EgoToM step 1 (moviepy ffmpeg_extract_subclip, stream copy) for sources that only
                 exist on Ego4D's AWS -> liyu/benchmarks/egotom/_stage1/<cond>/<cuid>_sub.mp4
  egotom         (rjob)    official step 1 for hdd1 sources + official step 2 (ffmpeg_resize to [480, 720]) for every
                 clip -> liyu/benchmarks/egotom/videos/<cond>/<cuid>_context.mp4; loops until stage 1 is complete
All sources are read through rangeproxy.RangeProxy (ffmpeg only talks to 127.0.0.1: the static ffmpeg build
segfaults on DNS lookups, and the proxy parallelises reads behind the per-connection-throttled kubebrain proxy).
Resumable: objects already in the bucket are skipped. Per-clip results -> logs/<mode>.jsonl.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
import boto3, imageio_ffmpeg
from botocore.config import Config
from rangeproxy import RangeProxy

D = os.path.dirname(os.path.abspath(__file__))
BUCKET = "safevlagent"
FF = imageio_ffmpeg.get_ffmpeg_exe()
h2 = boto3.session.Session(profile_name="h-hdd2").client(
    "s3", endpoint_url="http://hdd2.h.pjlab.org.cn:8060",
    config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 10, "mode": "standard"},
                  connect_timeout=20, read_timeout=120, max_pool_connections=64))
lock = threading.Lock()
PROXY = "http://httpproxy-headless.kubebrain.svc.pjlab.local:3128"


def say(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def log(mode, rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with lock, open(f"{D}/logs/{mode}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def listing(prefix):
    keys = {}
    for page in h2.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            keys[o["Key"]] = o["Size"]
    return keys


def duration(path):
    r = subprocess.run([FF, "-hide_banner", "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg rc {r.returncode}: {r.stderr[-400:]}")


def set_proxy(on):
    for k in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
        os.environ.pop(k, None)
    if on:
        os.environ.update(https_proxy=PROXY, http_proxy=PROXY,
                          no_proxy="localhost,127.0.0.1,.pjlab.org.cn,.h.pjlab.org.cn")


# ---------------------------------------------------------------- EgoTempo
def egotempo_one(rp, t, tmpdir):
    key = f"liyu/benchmarks/egotempo/clips/{t['clip']}"
    lu, tok = rp.register(t["src"])
    try:
        out = f"{tmpdir}/{t['clip']}"
        vf = ["-vf", "scale='if(gt(iw,ih),-2,540)':'if(gt(iw,ih),540,-2)'"] if t["scale540"] else []
        run([FF, "-hide_banner", "-loglevel", "error", "-ss", f"{t['start']:.3f}", "-i", lu, "-t", f"{t['end'] - t['start']:.3f}",
             "-map", "0:v:0", "-map", "0:a:0?", *vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-threads", "2", "-y", out])
        dur = duration(out)
        want = t["end"] - t["start"]
        if dur is None or abs(dur - want) > 1.0:
            raise RuntimeError(f"duration {dur} vs window {want:.2f}")
        h2.upload_file(out, BUCKET, key)
        os.remove(out)
        return {"clip": t["clip"], "status": "ok", "dur": dur, "src_kind": t["src_kind"]}
    finally:
        rp.release(tok)


# ---------------------------------------------------------------- EgoToM (official moviepy 1.0.3 calls)
def extract_subclip(src_url, t1, t2, target):
    """Official step 1: moviepy ffmpeg_extract_subclip (-ss t1 -i SRC -t (t2-t1) -map 0 -c copy). If the mp4 muxer rejects
    one of the source's data streams, retry with video+audio only (recorded as fallback)."""
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    try:
        ffmpeg_extract_subclip(filename=src_url, t1=t1, t2=t2, targetname=target)
        return None
    except Exception as e:
        if os.path.exists(target):
            os.remove(target)
        run([FF, "-y", "-ss", "%0.2f" % t1, "-i", src_url, "-t", "%0.2f" % (t2 - t1),
             "-map", "0:v", "-map", "0:a?", "-vcodec", "copy", "-acodec", "copy", target])
        return f"map0_failed: {str(e)[-160:]}"


def resize(sub, out):
    """Official step 2: moviepy ffmpeg_resize(video, output, size=[480, 720])."""
    from moviepy.video.io.ffmpeg_tools import ffmpeg_resize
    ffmpeg_resize(video=sub, output=out, size=[480, 720])


def egotom_group(rp, rows, tmpdir, mode, have_final, have_stage):
    """All context conditions of one clip, sequentially (they share source blocks in the proxy cache)."""
    res = []
    tok = None
    try:
        for t in rows:
            t1 = max(0.0, t["t1"])                       # a clip shorter than 30 s/5 s would give a negative start
            final = f"liyu/benchmarks/egotom/videos/{t['cond']}/{t['cuid']}_context.mp4"
            stage = f"liyu/benchmarks/egotom/_stage1/{t['cond']}/{t['cuid']}_sub.mp4"
            sub = f"{tmpdir}/{t['cond']}__{abs(hash(t['cuid']))}_sub.mp4"
            out = f"{tmpdir}/{t['cond']}__{abs(hash(t['cuid']))}_context.mp4"
            note = "t1 clamped to 0" if t["t1"] < 0 else None
            if mode == "egotom-stage1":
                if stage in have_stage or final in have_final:
                    continue
                if tok is None:
                    lu, tok = rp.register(t["src"])
                fb = extract_subclip(lu, t1, t["t2"], sub)
                h2.upload_file(sub, BUCKET, stage)
                res.append({"cuid": t["cuid"], "cond": t["cond"], "status": "staged", "fallback": fb, "note": note,
                            "sub_dur": duration(sub)})
                os.remove(sub)
                continue
            # mode == "egotom"
            if final in have_final:
                continue
            fb = None
            if t["src_kind"] == "hdd1_full":
                if tok is None:
                    lu, tok = rp.register(t["src"])
                fb = extract_subclip(lu, t1, t["t2"], sub)
            elif stage in have_stage:
                h2.download_file(BUCKET, stage, sub)
            else:
                res.append({"cuid": t["cuid"], "cond": t["cond"], "status": "waiting_stage1"})
                continue
            resize(sub, out)
            dur = duration(out)
            want = t["t2"] - t1
            if dur is None or not (want - 1.0 <= dur <= want + 6.0):   # stream-copied step 1 starts at a keyframe
                raise RuntimeError(f"{t['cuid']} {t['cond']}: duration {dur} vs window {want:.2f}")
            h2.upload_file(out, BUCKET, final)
            res.append({"cuid": t["cuid"], "cond": t["cond"], "status": "ok", "dur": dur, "src_kind": t["src_kind"],
                        "fallback": fb, "note": note})
            for p in (sub, out):
                if os.path.exists(p):
                    os.remove(p)
        return res
    finally:
        if tok:
            rp.release(tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["egotempo", "egotom-stage1", "egotom"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--proxy-workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N clips")
    ap.add_argument("--tasks", default=None, help="task file (default tasks_<dataset>.jsonl)")
    a = ap.parse_args()
    os.makedirs(f"{D}/logs", exist_ok=True)
    set_proxy(a.mode != "egotom")                      # rjob pods have no internet; openvla goes out via kubebrain
    rp = RangeProxy(workers=a.proxy_workers, log=say).start()
    tmpdir = tempfile.mkdtemp(prefix=f"egoclip_{a.mode}_", dir=os.environ.get("EGOCLIP_TMP", "/tmp"))
    ds = "egotempo" if a.mode == "egotempo" else "egotom"
    tasks = [json.loads(l) for l in open(a.tasks or f"{D}/tasks_{ds}.jsonl")]
    if a.limit and ds == "egotempo":
        tasks = tasks[:a.limit]
    if a.mode == "egotempo":
        have = listing("liyu/benchmarks/egotempo/clips/")
        todo = [t for t in tasks if f"liyu/benchmarks/egotempo/clips/{t['clip']}" not in have]
        say(f"egotempo: {len(tasks)} clips, {len(todo)} to cut, {a.workers} workers")
        ok = fail = 0

        def one(t):
            try:
                return egotempo_one(rp, t, tmpdir)
            except Exception as e:
                return {"clip": t["clip"], "status": "fail", "err": str(e)[:400]}
        with ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(one, todo):
                log(a.mode, r)
                ok += r["status"] == "ok"
                fail += r["status"] == "fail"
                if (ok + fail) % 20 == 0:
                    say(f"egotempo {ok + fail}/{len(todo)} ok={ok} fail={fail} fetched {rp.blocks.fetched / 1e9:.2f} GB")
        say(f"DONE egotempo ok={ok} fail={fail} fetched {rp.blocks.fetched / 1e9:.2f} GB")
        return
    groups = {}
    for t in tasks:
        if a.mode == "egotom-stage1" and t["src_kind"] != "aws_full":
            continue
        groups.setdefault(t["cuid"], []).append(t)
    if a.limit:
        groups = dict(list(groups.items())[:a.limit])
    npass = 0
    while True:
        npass += 1
        have_final = listing("liyu/benchmarks/egotom/videos/")
        have_stage = listing("liyu/benchmarks/egotom/_stage1/")
        stage_done = "liyu/benchmarks/egotom/_stage1/_DONE" in have_stage
        say(f"{a.mode} pass {npass}: {len(groups)} clips x conds; final {len(have_final)}, staged {len(have_stage)}")
        counts = {}

        def one(rows):
            try:
                return egotom_group(rp, rows, tmpdir, a.mode, have_final, have_stage)
            except Exception as e:
                return [{"cuid": rows[0]["cuid"], "status": "fail", "err": str(e)[:400]}]
        with ThreadPoolExecutor(a.workers) as ex:
            for res in ex.map(one, list(groups.values())):
                for r in res:
                    counts[r["status"]] = counts.get(r["status"], 0) + 1
                    if r["status"] != "waiting_stage1":
                        log(a.mode, r)
        say(f"{a.mode} pass {npass} result {counts}; fetched {rp.blocks.fetched / 1e9:.2f} GB")
        if a.mode == "egotom-stage1":
            if not counts.get("fail"):
                h2.put_object(Bucket=BUCKET, Key="liyu/benchmarks/egotom/_stage1/_DONE", Body=json.dumps(counts).encode())
                say("DONE egotom-stage1")
                return
            if npass >= 3:
                say(f"DONE egotom-stage1 with failures {counts}")
                return
            continue
        waiting, failed = counts.get("waiting_stage1", 0), counts.get("fail", 0)
        if not waiting and not failed:
            say("DONE egotom")
            return
        if stage_done and not counts.get("ok") and npass > 3:
            say(f"DONE egotom with leftovers {counts}")
            return
        time.sleep(300 if waiting else 60)


if __name__ == "__main__":
    main()
