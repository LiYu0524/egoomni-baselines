#!/usr/bin/env python3
"""Parallel, resumable, sha256-verified download of a ModelScope repo subtree through the pjlab proxy.
usage: ms_dl.py OWNER/MODEL SUBDIR OUTDIR [STREAMS] [FILE_GLOB]"""
import hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

repo, sub, out = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 24
only = sys.argv[5] if len(sys.argv) > 5 else ""
API = "https://www.modelscope.ai/api/v1/models"
CHUNK = 20_000_000
env = dict(os.environ)
for k in ("no_proxy", "NO_PROXY"):
    env[k] = "localhost,127.0.0.1,10.0.0.0/8,100.0.0.0/8,.pjlab.org.cn"

def curl(url, rng, dest):
    for attempt in range(6):
        r = subprocess.run(["curl", "-sL", "--max-time", "600", "-r", rng, "-o", dest, url], env=env)
        if r.returncode == 0 and os.path.getsize(dest) > 0:
            return True
        time.sleep(2 * (attempt + 1))
    return False

files = json.loads(subprocess.run(["curl", "-s", "--max-time", "60", f"{API}/{repo}/repo/files?Recursive=true"],
                                  env=env, capture_output=True, text=True).stdout)["Data"]["Files"]
todo = [f for f in files if f["Type"] == "blob" and f["Path"].startswith(sub) and (not only or only in f["Path"])]
print(f"{len(todo)} files, {sum(f['Size'] for f in todo)/1e9:.2f} GB → {out}", flush=True)
for f in sorted(todo, key=lambda x: -x["Size"]):
    rel = f["Path"][len(sub):].lstrip("/")
    dest = os.path.join(out, rel)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) == f["Size"]:
        print(f"skip {rel}", flush=True); continue
    url = f"{API}/{repo}/repo?Revision=master&FilePath={f['Path']}"
    t0 = time.time()
    parts = [(i, min(CHUNK, f["Size"] - i * CHUNK)) for i in range((f["Size"] + CHUNK - 1) // CHUNK)]
    with ThreadPoolExecutor(N) as ex:
        ok = list(ex.map(lambda p: curl(url, f"{p[0]*CHUNK}-{p[0]*CHUNK + p[1] - 1}", f"{dest}.part{p[0]}"), parts))
    assert all(ok), f"{rel}: {ok.count(False)} chunks failed"
    with open(dest + ".tmp", "wb") as o:
        for i, _ in parts:
            with open(f"{dest}.part{i}", "rb") as p:
                while (b := p.read(1 << 22)):
                    o.write(b)
            os.remove(f"{dest}.part{i}")
    h = hashlib.sha256()
    with open(dest + ".tmp", "rb") as p:
        while (b := p.read(1 << 22)):
            h.update(b)
    assert os.path.getsize(dest + ".tmp") == f["Size"], f"{rel}: size mismatch"
    if f["Sha256"] and h.hexdigest() != f["Sha256"]:
        raise AssertionError(f"{rel}: sha256 mismatch")
    os.replace(dest + ".tmp", dest)
    print(f"ok {rel} {f['Size']/1e6:.0f} MB in {time.time()-t0:.0f} s ({f['Size']/1e6/max(1,time.time()-t0):.1f} MB/s) sha ok", flush=True)
print("ALL DONE", flush=True)
