#!/usr/bin/env python3
"""Parallel, resumable HF repo download through the per-connection-throttled kubebrain proxy (openvla).

usage: hf_parallel_dl.py REPO REVISION OUT_DIR [CONNECTIONS]
Each LFS file is fetched as 8 MiB ranges over many connections into a preallocated file (os.pwrite); finished ranges
are recorded in <file>.parts so a restart resumes. Every LFS file is verified against HF's sha256 before the .parts
file is removed; small files are fetched whole.
"""
import hashlib, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests

REPO, REV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CONN = int(sys.argv[4]) if len(sys.argv) > 4 else 32
EP = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
CH = 8 << 20
tls = threading.local()


def sess():
    if not hasattr(tls, "s"):
        tls.s = requests.Session()
    return tls.s


def say(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


def get_range(url, a, b):
    for attempt in range(60):
        try:
            r = sess().get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=(30, 120))
            if r.status_code == 206 and len(r.content) == b - a + 1:
                return r.content
        except Exception:
            pass
        time.sleep(min(30, 2 ** min(attempt, 5)))
    raise IOError(f"range {a}-{b} of {url} failed")


def fetch_file(name, size, sha):
    url = f"{EP}/{REPO}/resolve/{REV}/{name}"
    dst = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not sha:
        for attempt in range(20):
            try:
                r = sess().get(url, timeout=(30, 120))
                if r.status_code == 200 and len(r.content) == size:
                    open(dst, "wb").write(r.content)
                    return
            except Exception:
                pass
            time.sleep(5)
        raise IOError(f"{name} failed")
    partf = dst + ".parts"
    if os.path.exists(dst) and not os.path.exists(partf) and os.path.getsize(dst) == size:
        return
    done = set(json.load(open(partf))) if os.path.exists(partf) else set()
    fd = os.open(dst, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, size)
    todo = [i for i in range((size + CH - 1) // CH) if i not in done]
    lock = threading.Lock()
    t0, got = time.time(), [0]

    def one(i):
        a = i * CH
        data = get_range(url, a, min(size, a + CH) - 1)
        os.pwrite(fd, data, a)
        with lock:
            done.add(i)
            got[0] += len(data)
            if len(done) % 25 == 0:
                json.dump(sorted(done), open(partf, "w"))
                say(f"{name}: {len(done)}/{(size + CH - 1) // CH} chunks, {got[0] / (time.time() - t0) / 1e6:.2f} MB/s")
    with ThreadPoolExecutor(CONN) as ex:
        list(ex.map(one, todo))
    os.fsync(fd)
    os.close(fd)
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for blk in iter(lambda: f.read(64 << 20), b""):
            h.update(blk)
    if h.hexdigest() != sha:
        os.remove(partf) if os.path.exists(partf) else None
        raise IOError(f"{name}: sha256 mismatch")
    if os.path.exists(partf):
        os.remove(partf)
    say(f"{name}: done, sha256 OK")


def main():
    info = requests.get(f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true", timeout=60).json() \
        if os.environ.get("HF_META_DIRECT") else requests.get(f"{EP}/api/models/{REPO}/revision/{REV}?blobs=true", timeout=60).json()
    files = [(s["rfilename"], s["size"], (s.get("lfs") or {}).get("sha256")) for s in info["siblings"]]
    say(f"{REPO}@{REV}: {len(files)} files, {sum(f[1] for f in files) / 1e9:.2f} GB, {CONN} connections")
    for name, size, sha in sorted(files, key=lambda f: f[1]):
        fetch_file(name, size, sha)
    say("ALL DONE")


if __name__ == "__main__":
    main()
