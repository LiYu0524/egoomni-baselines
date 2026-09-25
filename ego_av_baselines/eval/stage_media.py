#!/usr/bin/env python3
"""Copy benchmark media from s3://safevlagent/liyu/benchmarks/ to the pod's local disk (no internet needed).

usage: stage_media.py ROOT BENCH [BENCH ...]      BENCH in egotom, egotempo, egosound
  egotom   -> ROOT/egotom/videos/<cond>/<cuid>_context.mp4
  egotempo -> ROOT/egotempo/clips/<clip>.mp4
  egosound -> ROOT/egosound/{Ego4d,EgoBlind}/{videos,audios}/... (the two official zips, extracted)
A ROOT/.<bench>.ok marker makes it idempotent across jobs on the same node.
"""
import os, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.config import Config

B = "safevlagent"
s3 = boto3.session.Session(profile_name="h-hdd2").client(
    "s3", endpoint_url="http://hdd2.h.pjlab.org.cn:8060",
    config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=64))


def listing(prefix):
    out = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=prefix):
        out += [(o["Key"], o["Size"]) for o in page.get("Contents", [])]
    return out


def pull(pairs):
    def one(kv):
        key, dst, size = kv
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            return 0
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        s3.download_file(B, key, dst + ".part")
        os.replace(dst + ".part", dst)
        return size
    with ThreadPoolExecutor(32) as ex:
        return sum(ex.map(one, pairs))


def main():
    root, benches = sys.argv[1], sys.argv[2:]
    for b in benches:
        mark = os.path.join(root, f".{b}.ok")
        if os.path.exists(mark):
            print(f"[stage] {b}: already staged", flush=True)
            continue
        t = time.time()
        if b in ("egotom", "egotempo"):
            pre = {"egotom": "liyu/benchmarks/egotom/videos/", "egotempo": "liyu/benchmarks/egotempo/clips/"}[b]
            sub = {"egotom": "egotom/videos/", "egotempo": "egotempo/clips/"}[b]
            objs = [(k, os.path.join(root, sub, k[len(pre):]), s) for k, s in listing(pre)]
            n = pull(objs)
            print(f"[stage] {b}: {len(objs)} files, {n / 1e9:.2f} GB new, {time.time() - t:.0f}s", flush=True)
        elif b == "egosound":
            d = os.path.join(root, "egosound")
            zips = [(f"liyu/benchmarks/egosound/{z}", os.path.join(d, "_zip", z), s)
                    for z, s in (("Ego4d.zip", 3460662881), ("EgoBlind.zip", 35050881705))]
            pull(zips)
            for _, zp, _ in zips:
                with zipfile.ZipFile(zp) as z:
                    z.extractall(d)
                os.remove(zp)
            nv = sum(len(os.listdir(os.path.join(d, s, "videos"))) for s in ("Ego4d", "EgoBlind"))
            print(f"[stage] egosound: extracted, {nv} videos, {time.time() - t:.0f}s", flush=True)
        else:
            raise SystemExit(f"unknown bench {b}")
        open(mark, "w").write(time.strftime("%F %T"))


if __name__ == "__main__":
    main()
