#!/usr/bin/env python3
"""EgoSchema public Subset (500 five-way MCQ with answers) as LLaMAFactory datasets for the EgoAVU Qwen2.5-Omni models.
Prompt = lmms-eval egoschema: question, the five "A. ..." options on new lines, then
"Answer with the option's letter from the given choices directly."  Media = the EgoAVU training recipe on the proxies:
video "<proxy>.mp4#t=0,<dur>" (2 fps, <=64 frames, <=200,704 px) + 16 kHz FLAC audio when the source has audio.
Writes data/egoschema_subset_s{0,1}of2.jsonl, dataset_info.json, meta.json.   usage: build_data.py"""
import json
R = "/mnt/shared-storage-user/ai4good1-share/liyu/egoschema_eval_h"; P = "/ai4good1-shared/liyu/egoschema_eval_h/proxy"
qs = sorted(json.load(open(f"{R}/meta/subset.json")), key=lambda x: x["question_idx"])
media = json.load(open(f"{R}/meta/media_subset.json"))
shards = {f"egoschema_subset_s{k}of2": [] for k in range(2)}
for n, q in enumerate(qs):
    m = media[q["video_idx"]]; assert "error" not in m, q["video_idx"]
    prompt = q["question"] + "".join("\n" + o for o in q["option"]) + "\nAnswer with the option's letter from the given choices directly."
    gold = "ABCDE"[int(q["answer"])]
    row = {"messages": [{"role": "user", "content": ("<video><audio>" if m["has_audio"] else "<video>") + prompt},
                        {"role": "assistant", "content": gold}],
           "videos": [f"{P}/{q['video_idx']}.mp4#t=0,{m['proxy_duration']:.3f}"],
           "audios": [f"{P}/{q['video_idx']}.flac"] if m["has_audio"] else []}
    shards[f"egoschema_subset_s{n % 2}of2"].append((row, {"question_idx": q["question_idx"], "video_idx": q["video_idx"],
                                                          "gold": gold, "options": q["option"], "has_audio": m["has_audio"]}))
info, meta = {}, {}
for name, rows in shards.items():
    with open(f"{R}/data/{name}.jsonl", "w") as f:
        for row, _ in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta[name] = [m for _, m in rows]
    info[name] = {"file_name": f"{name}.jsonl", "formatting": "sharegpt",
                  "columns": {"messages": "messages", "videos": "videos", "audios": "audios"},
                  "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
json.dump(info, open(f"{R}/data/dataset_info.json", "w"), indent=2); json.dump(meta, open(f"{R}/data/meta.json", "w"))
print({k: len(v) for k, v in shards.items()}, "| with audio:", sum(m["has_audio"] for v in meta.values() for m in v))
