#!/usr/bin/env python3
"""Mirror HF dataset grooLegend/egoOmni (pinned revision) into s3://safevlagent/liyu/egoOmni/ — no lasting local copy.
Per file: download to a temp file on gpfs (HTTP Range resume), verify size + sha256 (HF LFS oid), upload, delete it.
Resumable: skips objects already in the bucket with the right size. Appends results to a manifest jsonl.
usage: mirror_egoomni_hf_to_s3.py REVISION [WORKERS]"""
import os, sys, json, time, hashlib, threading, concurrent.futures as cf
import requests, boto3
from botocore.config import Config

REPO, REV = "grooLegend/egoOmni", sys.argv[1]
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BUCKET, PREFIX = "safevlagent", "liyu/egoOmni/"
L = "/mnt/shared-storage-user/ai4good1-share/liyu"
TMP = f"{L}/.cache/egoomni_dl"; os.makedirs(TMP, exist_ok=True)
MAN = f"{L}/tools/h/logs/egoomni_mirror_manifest.jsonl"
os.environ.setdefault("AWS_SHARED_CREDENTIALS_FILE", f"{L}/.aws/credentials")
os.environ.setdefault("AWS_CONFIG_FILE", f"{L}/.aws/config")
s3 = boto3.session.Session(profile_name="h-hdd2").client(
    "s3", endpoint_url="http://hdd2.h.pjlab.org.cn:8060",
    config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 10, "mode": "adaptive"}, max_pool_connections=64))

meta = requests.get(f"https://huggingface.co/api/datasets/{REPO}/revision/{REV}?blobs=true", timeout=60).json()
files = [(x["rfilename"], x["size"], (x.get("lfs") or {}).get("sha256")) for x in meta["siblings"]]
have = {}
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREFIX):
    for o in page.get("Contents", []):
        have[o["Key"][len(PREFIX):]] = o["Size"]
todo = [f for f in files if have.get(f[0]) != f[1]]
tot_gb = sum(f[1] for f in files) / 1e9; todo_gb = sum(f[1] for f in todo) / 1e9
print(f"[mirror] rev {REV[:7]}: {len(files)} files {tot_gb:.2f} GB | in bucket {len(files)-len(todo)} | todo {len(todo)} ({todo_gb:.2f} GB) | workers {WORKERS}", flush=True)

ATTEMPTS = 25


def one(f):
    """Download with HTTP Range resume (a stalled big clip continues where it stopped), verify size + sha256, upload."""
    name, size, sha = f
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{name}"
    tmp = os.path.join(TMP, hashlib.md5(name.encode()).hexdigest())
    err = None
    for attempt in range(ATTEMPTS):
        try:
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if have > size:
                os.remove(tmp); have = 0
            h = hashlib.sha256()
            if have:
                with open(tmp, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 24), b""):
                        h.update(chunk)
            if have < size:
                with requests.get(url, stream=True, timeout=(30, 600), headers={"Range": f"bytes={have}-"} if have else {}) as r:
                    r.raise_for_status()
                    if have and r.status_code != 206:           # Range ignored -> restart this file
                        have, h = 0, hashlib.sha256()
                    with open(tmp, "ab" if have else "wb") as out:
                        for chunk in r.iter_content(1 << 20):
                            out.write(chunk); h.update(chunk)
            n = os.path.getsize(tmp)
            if n != size:
                raise IOError(f"size {n} != {size}")
            if sha and h.hexdigest() != sha:
                os.remove(tmp); raise IOError("sha256 mismatch (restarting file)")
            s3.upload_file(tmp, BUCKET, PREFIX + name)
            os.remove(tmp)
            return {"file": name, "size": size, "sha256": h.hexdigest(), "ok": True}
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(60, 5 * 2 ** min(attempt, 4)))
    return {"file": name, "size": size, "ok": False, "error": err}      # partial tmp kept for the next run to resume


t0, done, nok, gb_ok, lock = time.time(), 0, 0, 0.0, threading.Lock()
with cf.ThreadPoolExecutor(WORKERS) as ex, open(MAN, "a") as man:
    for fut in cf.as_completed([ex.submit(one, f) for f in todo]):
        res = fut.result(); done += 1
        man.write(json.dumps(res) + "\n"); man.flush()
        if res["ok"]:
            nok += 1; gb_ok += res["size"] / 1e9
        else:
            print("[mirror] FAIL", res["file"], res["error"], flush=True)
        if done % 50 == 0 or done == len(todo):
            el = time.time() - t0
            print(f"[mirror] {done}/{len(todo)} files, {gb_ok:.2f}/{todo_gb:.2f} GB ok, {gb_ok*1e3/max(el,1):.1f} MB/s, "
                  f"ETA {(todo_gb-gb_ok)/max(gb_ok/max(el,1),1e-9)/3600:.1f} h", flush=True)
print(f"[mirror] finished: {nok}/{len(todo)} ok, {len(todo)-nok} failed, {time.time()-t0:.0f} s", flush=True)
