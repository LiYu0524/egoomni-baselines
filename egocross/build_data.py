#!/usr/bin/env python3
"""EgoCross closed-set (957 MCQ, 4 domains) as LLaMAFactory datasets for the EgoAVU Qwen2.5-Omni models.
Protocol = the benchmark's own EgoCross/eval_os_closedset.py: the frame list is the video; sampling fps 0.5, except
CholecTrack20 .png frames at 1.0; identical prompt text; greedy decoding; max_pixels 360*480 (set in make_config.py).
41 questions list frame files that are absent from the released testbed (also absent from its ModelScope file list); each
missing frame is replaced by its nearest existing neighbour so frame count and timing are unchanged; flagged `incomplete`.
Writes data/{egocross_fps05_s0of2,egocross_fps05_s1of2,egocross_fps10}.jsonl, dataset_info.json, meta.json."""
import json, os
E = "/mnt/shared-storage-user/liyu/egocross"; T = E + "/eval/data/EgoCross"
OUT = "/mnt/shared-storage-user/ai4good1-share/liyu/egocross_eval_h/data"; os.makedirs(OUT, exist_ok=True)
ans = json.load(open(E + "/codabench_v2/reference_data/answers.json"))


def prompt(it, sampling_fps):     # verbatim from eval_os_closedset.py (incl. its missing separators)
    options_text = it.get("options", [])
    options_str = "\n\nOptions:\n" + "\n".join(options_text) if options_text else ""
    return ("Please carefully read the question and its options, then select the most appropriate answer. "
            f"Question: {it['question_text']}{options_str}"
            f"The original FPS of the video is {it.get('original_video_fps', 30.0)}. This image set is obtained by sampling at {sampling_fps} fps."
            "Respond in JSON format with two fields: 'prediction' (the correct option letter: A, B, C, or D) and 'reason' (a brief explanation of your choice). "
            "Do not include any other content.\n\n"
            "Example response:\n"
            "{\n"
            "    \"prediction\": \"B\",\n"
            "    \"reason\": \"Paris is the capital city of France.\"\n"
            "}\n")


groups = {"egocross_fps05_s0of2": [], "egocross_fps05_s1of2": [], "egocross_fps10": []}
n05 = 0
for it in ans:
    paths = [T + p for p in it["video_path"]]
    ok = [os.path.exists(p) for p in paths]
    assert any(ok), it["question_id"]
    frames = [p if ok[i] else paths[min((k for k in range(len(paths)) if ok[k]), key=lambda k: (abs(k - i), k))]
              for i, p in enumerate(paths)]
    fps = 1.0 if it["dataset"] == "CholecTrack20" and os.path.splitext(it["video_path"][0])[1].lower() == ".png" else 0.5
    row = {"messages": [{"role": "user", "content": "<video>" + prompt(it, fps)},
                        {"role": "assistant", "content": it["correct_option_letter"]}], "videos": [frames]}
    m = {"question_id": it["question_id"], "gold": it["correct_option_letter"], "dataset": it["dataset"],
         "category": it.get("primary_category"), "n_frames": len(paths), "incomplete": not all(ok), "fps": fps}
    if fps == 1.0:
        g = "egocross_fps10"
    else:
        g = f"egocross_fps05_s{n05 % 2}of2"; n05 += 1
    groups[g].append((row, m))

info, meta = {}, {}
for g, rows in groups.items():
    with open(f"{OUT}/{g}.jsonl", "w") as f:
        for row, _ in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta[g] = [m for _, m in rows]
    info[g] = {"file_name": f"{g}.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages", "videos": "videos"},
               "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
json.dump(info, open(f"{OUT}/dataset_info.json", "w"), indent=2)
json.dump(meta, open(f"{OUT}/meta.json", "w"))
print({g: len(r) for g, r in groups.items()}, "| incomplete:", sum(m["incomplete"] for ms in meta.values() for m in ms))
