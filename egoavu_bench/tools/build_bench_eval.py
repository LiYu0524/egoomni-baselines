#!/usr/bin/env python3
"""EgoAVU-Bench inference set, in OFFICIAL BENCH ORDER, with a verified bench_idx per row.

Source of truth: data/EgoAVU_data/egoavu_bench_combined.csv (3,976 rows; bench_idx = 0-based CSV row).
Media/prompt rows are taken verbatim from datasets_win/egoavu_test_win.jsonl (the training-identical
window rows: proxy262k + "#t=a,b" fragment, "<video><audio>" + question). That file is ordered by
(video_id sorted, CSV order within video) with stripped text; this script replays that ordering to
recover bench_idx and asserts every field matches before writing anything.
Outputs (data/): egoavu_bench_eval.jsonl, egoavu_bench_eval_s{k}of{n}.jsonl, egoavu_bench_smoke.jsonl,
dataset_info.json, build_summary.json."""
import csv, json, os, sys
from collections import defaultdict, Counter

csv.field_size_limit(sys.maxsize)
R = "/ai4good1-shared/liyu/egoavu"
OUT = R + "/bench_infer/data"
NSHARDS = [int(x) for x in os.environ.get("NSHARDS", "2,3,4").split(",")]
SHORT = {"Audio-Visual Segment Narration": "avsn", "Audio-Visual Dense Narration": "avdn",
         "Sound Source Association": "ssa", "Temporal Reasoning": "tr",
         "Audio Visual Hallucination (Action)": "avh_action", "Audio Visual Hallucination (Object)": "avh_object",
         "Audio Visual Hallucination (Sound)": "avh_sound"}

bench = list(csv.DictReader(open(R + "/data/EgoAVU_data/egoavu_bench_combined.csv", newline="", encoding="utf-8")))
assert len(bench) == 3976, len(bench)
by_vid = defaultdict(list)
for i, r in enumerate(bench):
    q, a = (r["question"] or "").strip(), (r["answer"] or "").strip()
    assert q and a, f"blank q/a at bench row {i}"
    by_vid[r["video_id"]].append(i)
order = [i for v in sorted(by_vid) for i in by_vid[v]]          # == build_window_groups.load() ordering
win = [json.loads(l) for l in open(R + "/datasets_win/egoavu_test_win.jsonl")]
assert len(win) == len(order) == 3976

rows = []
for i, w in zip(order, win):
    b = bench[i]
    user, asst = w["messages"][0]["content"], w["messages"][1]["content"]
    prefix = "<video><audio>" if w["audios"] else "<video>"
    assert user == prefix + b["question"].strip(), (i, user[:80])
    assert asst == b["answer"].strip(), i
    assert w["category"] == b["category"].strip(), i
    assert f"/{b['video_id']}.mp4#t=" in w["videos"][0], i
    rows.append(dict(w, bench_idx=i, video_id=b["video_id"], start_time=float(b["start_time"]),
                     end_time=float(b["end_time"]), category_short=SHORT[w["category"]]))
rows.sort(key=lambda r: r["bench_idx"])
assert [r["bench_idx"] for r in rows] == list(range(3976))

os.makedirs(OUT, exist_ok=True)
def dump(name, rs):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
dump("egoavu_bench_eval", rows)
names = ["egoavu_bench_eval"]
for n in NSHARDS:            # round-robin in bench order -> every shard has the same category mix
    for k in range(n):
        dump(f"egoavu_bench_eval_s{k}of{n}", rows[k::n]); names.append(f"egoavu_bench_eval_s{k}of{n}")
# smoke: 2 per category + the 4 shortest windows + both video-only rows + the 2 longest windows
dur = lambda r: r["end_time"] - r["start_time"]
pick = {}
for c in SHORT:
    for r in [r for r in rows if r["category"] == c][:2]: pick[r["bench_idx"]] = r
for r in sorted(rows, key=dur)[:4] + [r for r in rows if not r["audios"]] + sorted(rows, key=dur)[-2:]:
    pick[r["bench_idx"]] = r
smoke = [pick[k] for k in sorted(pick)]
dump("egoavu_bench_smoke", smoke); names.append("egoavu_bench_smoke")

cols = {"messages": "messages", "videos": "videos", "audios": "audios"}
tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
json.dump({n: {"file_name": f"{n}.jsonl", "formatting": "sharegpt", "columns": cols, "tags": tags} for n in names},
          open(f"{OUT}/dataset_info.json", "w"), indent=2)
summary = {"rows": len(rows), "videos": len(by_vid), "with_audio": sum(bool(r["audios"]) for r in rows),
           "categories": Counter(r["category"] for r in rows), "smoke_rows": len(smoke),
           "shortest_windows_s": sorted(round(dur(r), 3) for r in rows)[:6], "files": names}
json.dump(summary, open(f"{OUT}/build_summary.json", "w"), indent=2); print(json.dumps(summary, indent=1))
