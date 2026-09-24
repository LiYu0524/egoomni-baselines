#!/usr/bin/env python3
"""Turn a thinker-only Qwen2.5-Omni checkpoint (config model_type qwen2_5_omni_thinker, tensor names without the `thinker.`
prefix — e.g. groo_legend/ckpts ckpt_fft_epoch2) into the full-Omni *thinker view* every EgoAVU run loads
(models/Qwen2.5-Omni-7B-thinker-train: model_type qwen2_5_omni, enable_audio_output false, tensors `thinker.*`).

Weights are never decoded: each shard gets a renamed safetensors header and its data section is copied byte-for-byte; the
written file is re-read and its data-section sha256 must equal the source's. Before anything is written: every source tensor
must map to a base `thinker.*` tensor with the same dtype/shape and every base `thinker.*` tensor must be covered.
While copying, each tensor is compared with the base weights (bytes) → per-module "changed" report (was the module trained?).
config.json = the base view's (so the architecture/processor settings match every other evaluated model); tokenizer/processor
files and spk_dict.pt are symlinked from the base view. The source's config/tokenizer/processor files are diffed against the
base and reported — review that report before using the output.
usage: convert_thinker_ckpt.py SRC_DIR BASE_VIEW_DIR OUT_DIR"""
import hashlib, json, os, shutil, struct, sys

CHUNK = 64 << 20


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return 8 + n, h                                     # (absolute start of the data section, header dict)


def group(name):                                        # module group of a thinker tensor name (without "thinker.")
    p = name.split(".")
    return ".".join(p[:2]) if p[0] == "model" else p[0]


def diff(a, b, path=""):                                # recursive config diff -> list of (path, a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}.{k}" if path else k)
        return out
    return [] if a == b else [(path, a, b)]


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def main(src, base, out):
    sidx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    bidx = json.load(open(f"{base}/model.safetensors.index.json"))["weight_map"]
    bt = {}                                             # base "thinker.X" -> (file, dtype, shape, abs_start, abs_end)
    for fn in sorted({v for k, v in bidx.items() if k.startswith("thinker.")}):
        d0, h = header(f"{base}/{fn}")
        for k, v in h.items():
            if k.startswith("thinker."):
                bt[k] = (fn, v["dtype"], v["shape"], d0 + v["data_offsets"][0], d0 + v["data_offsets"][1])
    shards = sorted(set(sidx.values()))
    sh = {fn: header(f"{src}/{fn}") for fn in shards}
    names = {k for fn in shards for k in sh[fn][1] if k != "__metadata__"}
    assert names == set(sidx), "source index and shard headers disagree"
    missing = sorted(set(bt) - {"thinker." + k for k in names})
    extra = sorted(k for k in names if "thinker." + k not in bt)
    bad = sorted(k for fn in shards for k, v in sh[fn][1].items()
                 if k != "__metadata__" and "thinker." + k in bt and bt["thinker." + k][1:3] != (v["dtype"], v["shape"]))
    print(f"source tensors {len(names)} | base thinker tensors {len(bt)} | missing {len(missing)} extra {len(extra)} "
          f"dtype/shape mismatch {len(bad)}", flush=True)
    if missing or extra or bad:
        print("missing:", missing[:8], "\nextra:", extra[:8], "\nmismatch:", bad[:8]); sys.exit(1)

    os.makedirs(out, exist_ok=True)
    wm, total, stats, bfiles = {}, 0, {}, {}
    for fn in shards:
        d0, h = sh[fn]
        meta = h.get("__metadata__")
        ts = sorted(((k, v) for k, v in h.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0])
        nh = ({"__metadata__": meta} if meta is not None else {}) | {"thinker." + k: v for k, v in ts}
        hb = json.dumps(nh, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        size = ts[-1][1]["data_offsets"][1]
        assert ts[0][1]["data_offsets"][0] == 0 and os.path.getsize(f"{src}/{fn}") == d0 + size, f"{fn}: unexpected layout"
        hs, tmp = hashlib.sha256(), f"{out}/{fn}.tmp"
        with open(f"{src}/{fn}", "rb") as fi, open(tmp, "wb") as fo:
            fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
            pos = 0
            for k, v in ts:
                a, e = v["data_offsets"]
                assert a == pos, f"{fn}: gap before {k}"
                bfn, _, _, ba, be = bt["thinker." + k]
                bf = bfiles.get(bfn) or bfiles.setdefault(bfn, open(f"{base}/{bfn}", "rb"))
                fi.seek(d0 + a); bf.seek(ba); same, left = True, e - a
                while left:
                    x = fi.read(min(CHUNK, left)); assert x, f"{fn}: short read"
                    hs.update(x); fo.write(x); left -= len(x)
                    if same and bf.read(len(x)) != x:
                        same = False
                s = stats.setdefault(group(k), [0, 0, 0, 0])
                s[0] += 1; s[1] += (not same); s[2] += e - a; s[3] += (e - a) * (not same)
                wm["thinker." + k] = fn; total += e - a; pos = e
        d1, h2 = header(tmp)                            # re-read what was written
        assert set(h2) - {"__metadata__"} == {"thinker." + k for k, _ in ts}
        hw = hashlib.sha256()
        with open(tmp, "rb") as f:
            f.seek(d1)
            for x in iter(lambda: f.read(CHUNK), b""):
                hw.update(x)
        assert hw.hexdigest() == hs.hexdigest(), f"{fn}: data section changed in the copy"
        os.replace(tmp, f"{out}/{fn}")
        print(f"{fn}: {len(ts)} tensors, {size / 1e9:.2f} GB, data sha256 {hs.hexdigest()[:16]} verified", flush=True)
    for f in bfiles.values():
        f.close()
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))},
              open(f"{out}/model.safetensors.index.json", "w"), indent=2)

    print("\nper-module comparison with the base weights (tensors changed / tensors, GB changed / GB):")
    for g, (n, c, b, bc) in sorted(stats.items()):
        print(f"  {g:28s} {c:5d}/{n:<5d} {bc / 1e9:6.2f}/{b / 1e9:.2f}")

    shutil.copyfile(f"{base}/config.json", f"{out}/config.json")
    for fn in sorted(os.listdir(base)):
        p = f"{base}/{fn}"
        if fn == "config.json" or fn.startswith("model") or not os.path.islink(p):
            continue
        if not os.path.lexists(f"{out}/{fn}"):
            os.symlink(os.readlink(p), f"{out}/{fn}")
    json.dump({"source": os.path.abspath(src), "base_view": os.path.abspath(base),
               "note": "thinker-only checkpoint re-headered to the full-Omni thinker view (tensor bytes unchanged); "
                       "config/tokenizer/processor = base view"}, open(f"{out}/CONVERTED_FROM.json", "w"), indent=1)

    print("\nsource config vs base thinker_config (differences):")
    for p, a, b in diff(json.load(open(f"{src}/config.json")), json.load(open(f"{base}/config.json"))["thinker_config"]):
        print(f"  {p}: source={json.dumps(a)[:120]} base={json.dumps(b)[:120]}")
    print("\nsource non-weight files vs base (sha256):")
    for fn in sorted(os.listdir(src)):
        if fn.startswith("model") or fn.endswith((".part", ".tmp")) or ".part" in fn:
            continue
        other = f"{base}/{fn}"
        state = "absent in base" if not os.path.exists(other) else ("identical" if file_sha(f"{src}/{fn}") == file_sha(other) else "DIFFERENT")
        print(f"  {fn:32s} {state}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
