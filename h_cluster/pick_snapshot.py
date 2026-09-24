#!/usr/bin/env python3
"""Print the first HF-cache snapshot of a model whose index lists only shards that exist (complete download).
usage: pick_snapshot.py /path/to/models--Org--Name"""
import glob, json, os, sys
for snap in sorted(glob.glob(os.path.join(sys.argv[1], "snapshots", "*"))):
    idx = os.path.join(snap, "model.safetensors.index.json")
    if not all(os.path.exists(os.path.join(snap, f)) for f in ("config.json", "tokenizer.json", "tokenizer_config.json")) or not os.path.exists(idx):
        continue
    shards = set(json.load(open(idx))["weight_map"].values())
    if all(os.path.exists(os.path.join(snap, s)) for s in shards):
        print(snap); sys.exit(0)
sys.exit("no complete snapshot under " + sys.argv[1])
