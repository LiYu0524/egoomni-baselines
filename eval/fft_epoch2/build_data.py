#!/usr/bin/env python
"""LLaMAFactory datasets for groo ckpt_fft_epoch2 (full fine-tune) on the egoOmni bench (original 3765 items + restored_v3 136).
Rows mirror the LoRA's training format: first user turn "<video><audio>" + question (or "<video>" for clips without audio),
video as "<clip>.mp4#t=0,<dur>" (the training code path: time-window frame counting + identical 2 fps / ≤64 sampling),
audio as 16 kHz mono FLAC. Question text/instructions = the harness's prompts.py (same as every other baseline).
  build_data.py gold K          → data/gold_s{k}of{K}.jsonl + index/gold_s{k}of{K}.json (item-level shards, balanced)
  build_data.py self K k R      → data/self_r{R}_s{k}of{K}.jsonl from this shard's earlier outputs (model's own history)
The last assistant message of every row is the gold answer of the asked turn (LLaMAFactory's label, used to verify the join)."""
import json, os, sys
sys.path.insert(0, "/ai4good1-shared/liyu/egoOmni_baselines/eval")
from egoomni_eval.data import build_item_plans, load_items, load_clips_meta, messages_for, read_jsonl

R = "/ai4good1-shared/liyu/egoOmni_baselines/eval/fft_epoch2"
SRC = [("/ai4good1-shared/liyu/egoOmni/test/qa.json", "/ai4good1-shared/liyu/egoOmni_baselines/eval/clips_meta.json"),
       ("/ai4good1-shared/liyu/egoOmni/test/qa_restored_v3.json", "/ai4good1-shared/liyu/egoOmni_baselines/eval/clips_meta_restored_v3.json")]


def plans():
    out = []
    for qa, cm in SRC:
        out += build_item_plans(load_items(qa), load_clips_meta(cm))
    return out


def shards(ps, K):
    gens = lambda p: p.n_turns + (p.n_turns - 1 if p.multi else 0)
    ps = sorted(ps, key=lambda p: (-gens(p), p.item_id)); load, b = [0] * K, [[] for _ in range(K)]
    for p in ps:
        i = load.index(min(load)); b[i].append(p); load[i] += gens(p)
    return [sorted(x, key=lambda p: p.item_id) for x in b]


def row(p, k, hist):
    msgs = messages_for(p, k, hist)
    pre = "<video><audio>" if p.has_audio else "<video>"
    msgs[0] = dict(msgs[0], content=pre + msgs[0]["content"])
    msgs.append({"role": "assistant", "content": p.turns[k - 1]["gold"]})
    audio = [os.path.join(R, "audio16k", p.clip_rel[:-4] + ".flac")] if p.has_audio else []
    return {"messages": msgs, "videos": [f"{p.clip}#t=0,{p.duration:.3f}"], "audios": audio}


def register(names):
    os.makedirs(f"{R}/data", exist_ok=True)
    info_p = f"{R}/data/dataset_info.json"
    info = json.load(open(info_p)) if os.path.exists(info_p) else {}
    for name in names:
        info[name] = {"file_name": f"{name}.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages", "videos": "videos", "audios": "audios"},
                      "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
    json.dump(info, open(info_p + ".tmp", "w"), indent=1); os.replace(info_p + ".tmp", info_p)


def write(name, rows, index):
    os.makedirs(f"{R}/data", exist_ok=True); os.makedirs(f"{R}/index", exist_ok=True)
    with open(f"{R}/data/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(index, open(f"{R}/index/{name}.json", "w"))
    print(name, len(rows))


if __name__ == "__main__":
    mode, K = sys.argv[1], int(sys.argv[2])
    sh = shards(plans(), K)
    if mode == "gold":           # register every dataset of the run up front: parallel jobs never rewrite dataset_info.json
        register([f"gold_s{k}of{K}" for k in range(K)] + [f"self_r{n}_s{k}of{K}" for n in (2, 3, 4) for k in range(K)])
        for k in range(K):
            rows, idx = [], []
            for p in sh[k]:
                for t in range(1, p.n_turns + 1):
                    rows.append(row(p, t, [p.turns[j]["gold"] for j in range(t - 1)])); idx.append([p.item_id, t, "gold"])
            write(f"gold_s{k}of{K}", rows, idx)
    else:
        k, Rn = int(sys.argv[3]), int(sys.argv[4])
        own = {}                         # (item, turn) -> model answer under the self protocol (round 1 = gold-protocol round 1)
        for r in read_jsonl(f"{R}/out/s{k}of{K}.jsonl"):
            if r["protocol"] == "self" or (r["protocol"] == "gold" and r["turn_idx"] == 1):
                own[(r["item_id"], r["turn_idx"])] = r["pred"]
        rows, idx = [], []
        for p in sh[k]:
            if p.multi and p.n_turns >= Rn:
                rows.append(row(p, Rn, [own.get((p.item_id, j + 1)) or "" for j in range(Rn - 1)])); idx.append([p.item_id, Rn, "self"])
        write(f"self_r{Rn}_s{k}of{K}", rows, idx)
