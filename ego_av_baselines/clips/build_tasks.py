#!/usr/bin/env python3
"""Build EgoTempo / EgoToM clip tasks (with 7-day presigned source URLs) and publish the annotations.

Run on openvla (it holds the ego4d + h-oss credentials; nothing secret is copied to shared storage — the task
files carry only time-limited presigned URLs and are chmod 600).

Sources
  EgoTempo: Ego4D video_540ss (what the official toolkit trims), fetched from Ego4D's AWS buckets.
  EgoToM:   Ego4D full_scale ("original Ego4D video" in the official script); hdd1 copy when we have it, else AWS.
Outputs (tasks_*.jsonl next to this file):
  egotempo: {clip, uid, start, end, src, src_kind, scale540}
  egotom:   {cuid, vuid, cond, t1, t2, src, src_kind}    one row per (clip, context condition)
"""
import csv, json, os, re, subprocess
import boto3
from botocore.config import Config

D = os.path.dirname(os.path.abspath(__file__))
P = "/root/egobench_probe"
EXP = 7 * 24 * 3600 - 600
BUCKET = "safevlagent"

e4 = boto3.Session(profile_name="ego4d").client("s3", region_name="us-east-1")
h1 = boto3.Session(profile_name="h-oss").client("s3", endpoint_url="http://hdd1.h.pjlab.org.cn:8060", region_name="us-east-1",
                                             config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"))
h2 = boto3.Session(profile_name="h-hdd2").client("s3", endpoint_url="http://hdd2.h.pjlab.org.cn:8060",
                                              config=Config(s3={"addressing_style": "path"}))
full = {r["video_uid"]: r for r in csv.DictReader(open(f"{P}/full_scale_manifest_v2_1.csv"))}
ss540 = {r["video_uid"]: r for r in csv.DictReader(open(f"{P}/video_540ss_manifest.csv"))}
in_hdd1 = {l.strip() for l in open(f"{P}/egoavu_video_ids.txt") if l.strip()}


_regional, _bucket_region = {}, {}


def bucket_region(b):
    """S3 names a bucket's region in the x-amz-bucket-region header, even on an anonymous 403."""
    if b not in _bucket_region:
        import requests
        r = requests.head(f"https://{b}.s3.amazonaws.com", timeout=60)
        _bucket_region[b] = r.headers.get("x-amz-bucket-region", "us-east-1")
    return _bucket_region[b]


def aws(loc):
    """Presign with the bucket's own regional endpoint: Ego4D buckets live in several regions (some opt-in, e.g.
    eu-south-1) where a URL signed for the global endpoint / us-east-1 is rejected."""
    b, k = loc[5:].split("/", 1)
    region = bucket_region(b)
    if region not in _regional:
        _regional[region] = boto3.Session(profile_name="ego4d").client(
            "s3", region_name=region, endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
    return _regional[region].generate_presigned_url("get_object", Params={"Bucket": b, "Key": k}, ExpiresIn=EXP)


def aws_ok(loc):
    b, k = loc[5:].split("/", 1)
    try:
        e4.head_object(Bucket=b, Key=k)
        return True
    except Exception:
        return False


def hdd1(uid):
    return h1.generate_presigned_url("get_object", ExpiresIn=EXP, Params={
        "Bucket": "ai4good-h-hdd-1", "Key": f"egoavu/raw/ego4d/v2_1/full_scale/{uid}.mp4"})


def full_src(uid):
    return (hdd1(uid), "hdd1_full") if uid in in_hdd1 else (aws(full[uid]["canonical_s3_location"]), "aws_full")


def put(key, path=None, body=None):
    if path:
        h2.upload_file(path, BUCKET, key)
    else:
        h2.put_object(Bucket=BUCKET, Key=key, Body=body)


def egotempo():
    ann = json.load(open(f"{P}/egotempo_openQA.json"))
    rows, seen = [], set()
    for a in ann["annotations"]:
        cid = a["clip_id"][:-4] if a["clip_id"].endswith(".mp4") else a["clip_id"]
        parts = cid.split("_")
        uid, s, e = "_".join(parts[:-2]), float(parts[-2]), float(parts[-1])
        name = f"{uid}_{s}_{e}.mp4"                      # exactly the official notebook's clip_name
        if name in seen:
            continue
        seen.add(name)
        rows.append({"clip": name, "uid": uid, "start": s, "end": e})
    by_uid = {}
    for r in rows:
        by_uid.setdefault(r["uid"], []).append(r)
    for uid, rs in by_uid.items():
        loc = ss540[uid]["s3_path"]
        if aws_ok(loc):
            src, kind, sc = aws(loc), "aws_540ss", False
        else:                                           # 540ss object missing/forbidden: full_scale scaled to 540p
            src, kind = full_src(uid)
            sc = True
        for r in rs:
            r.update(src=src, src_kind=kind, scale540=sc)
    put("liyu/benchmarks/egotempo/egotempo_openQA.json", path=f"{P}/egotempo_openQA.json")
    return rows


def egotom():
    base = "https://raw.githubusercontent.com/facebookresearch/EgoToM/main"
    files = ["egotom/all_prompts.json", "egotom_paper/all_prompts.json", "code/generate_video_context.py",
             "code/vlm_evaluate.py", "code/utils.py", "config/VLMeval/run_evaluation_multiexp.yaml", "README.md", "LICENSE.md"]
    files += [f"{d}/egotom_{q}_shuffled.csv" for d in ("egotom", "egotom_paper") for q in ("actions", "belief", "goal")]
    os.makedirs(f"{P}/egotom_repo", exist_ok=True)
    for f in files:
        dst = f"{P}/egotom_repo/{f}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(["curl", "-sfL", "--retry", "5", "-o", dst, f"{base}/{f}"], check=True)
        put(f"liyu/benchmarks/egotom/{f}", path=dst)
    seen, rows = set(), []
    for q in ("goal", "belief", "actions"):             # same concat order + drop_duplicates as the official script
        for r in csv.DictReader(open(f"{P}/egotom_repo/egotom/egotom_{q}_shuffled.csv")):
            k = (r["vuid"], r["cuid"], r["clip_start_time"], r["clip_end_time"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(k)
    srcs, out = {}, []
    for vuid, cuid, s, e in rows:
        if vuid not in srcs:
            srcs[vuid] = full_src(vuid)
        src, kind = srcs[vuid]
        s, e = float(s), float(e)
        for cond, t1 in (("fullcontext", s), ("last30sec", e - 30.0), ("last5sec", e - 5.0)):
            out.append({"cuid": cuid, "vuid": vuid, "cond": cond, "t1": t1, "t2": e, "src": src, "src_kind": kind})
    return out


def dump(name, rows):
    path = f"{D}/tasks_{name}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.chmod(path, 0o600)
    kinds = {}
    for r in rows:
        kinds[r["src_kind"]] = kinds.get(r["src_kind"], 0) + 1
    print(f"{name}: {len(rows)} tasks {kinds}")


if __name__ == "__main__":
    dump("egotempo", egotempo())
    dump("egotom", egotom())
