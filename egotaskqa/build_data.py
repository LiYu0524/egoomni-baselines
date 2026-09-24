#!/usr/bin/env python3
"""EgoTaskQA test splits (direct 8,783 + indirect 10,153 open short-answer questions) as LLaMAFactory datasets for the
Qwen2.5-Omni models. Prompt: the question, then "Answer the question using a single word or phrase." (short-answer VQA
convention). Media: the EgoAVU training recipe on the proxies (video "<proxy>.mp4#t=0,<dur>", 2 fps, <=64 frames,
<=200,704 px; 16 kHz FLAC audio only if the clip has sound).  Each split is sharded 4-way by list position.
Writes data/etq_{direct,indirect}_s{k}of4.jsonl, dataset_info.json, meta.json.   usage: build_data.py"""
import json
R = "/mnt/shared-storage-user/ai4good1-share/liyu/egotaskqa_eval_h"; P = "/ai4good1-shared/liyu/egotaskqa_eval_h/proxy"
media = json.load(open(f"{R}/meta/media.json"))
info, meta = {}, {}
for split in ("direct", "indirect"):
    qs = sorted(json.load(open(f"{R}/meta/qa_{split}_test_qas.json")), key=lambda q: q["question_id"])
    shards = {k: [] for k in range(4)}
    for n, q in enumerate(qs):
        m = media[q["interval"]]; assert "error" not in m, q["interval"]
        stem = f"{P}/{q['interval'].replace('|', '__')}"
        row = {"messages": [{"role": "user", "content": ("<video><audio>" if m["has_audio"] else "<video>") + q["question"]
                             + "\nAnswer the question using a single word or phrase."},
                            {"role": "assistant", "content": q["answer"]}],
               "videos": [f"{stem}.mp4#t=0,{m['proxy_duration']:.3f}"], "audios": [f"{stem}.flac"] if m["has_audio"] else []}
        shards[n % 4].append((row, {k: q[k] for k in ("question_id", "question", "answer", "type", "category", "semantic",
                                                      "structural", "step", "reasoning_type", "interval")}))
    for k, rows in shards.items():
        name = f"etq_{split}_s{k}of4"
        with open(f"{R}/data/{name}.jsonl", "w") as f:
            for row, _ in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta[name] = [m for _, m in rows]
        info[name] = {"file_name": f"{name}.jsonl", "formatting": "sharegpt",
                      "columns": {"messages": "messages", "videos": "videos", "audios": "audios"},
                      "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
json.dump(info, open(f"{R}/data/dataset_info.json", "w"), indent=2); json.dump(meta, open(f"{R}/data/meta.json", "w"))
print({k: len(v) for k, v in meta.items()}, "| clips with audio:", sum(bool(m.get("has_audio")) for m in media.values()))
