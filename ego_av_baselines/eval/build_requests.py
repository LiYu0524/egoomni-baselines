#!/usr/bin/env python3
"""Build model-agnostic request files for EgoToM, EgoTempo and EgoSound (run on openvla).

requests/<bench>.jsonl rows: {id, bench, video, audio?, system?, user, max_new_tokens, nframes, frame_mode, meta}
  video/audio are relative to the media root that stage_media.py fills on the pod's local disk.
  EgoToM   - official prompts (egotom/all_prompts.json, system + user), every question type x context condition
             (fullcontext / last30sec / last5sec), official frame selection 'uniform24' (end-aligned).
  EgoTempo - the official toolkit's question wrapper (gemini_eval.ipynb), 32 uniform frames.
  EgoSound - the question as-is (as in the official inference scripts), 32 uniform frames for frame-based models.
Ground truth stays out of the request files (scorers read the original annotations).
"""
import json, os

P = "/root/egobench_probe"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests")
os.makedirs(OUT, exist_ok=True)


def dump(name, rows):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))


def egotom():
    prompts = json.load(open(f"{P}/egotom_repo/egotom/all_prompts.json"))
    paper = json.load(open(f"{P}/egotom_repo/egotom_paper/all_prompts.json"))
    rows = []
    for cond in ("fullcontext", "last30sec", "last5sec"):
        for q in ("goal", "belief", "actions"):
            for cuid, pr in prompts[q].items():
                rows.append({"id": f"{cond}|{q}|{cuid}", "bench": "egotom", "video": f"egotom/videos/{cond}/{cuid}_context.mp4",
                             "system": pr["system"], "user": pr["user"], "max_new_tokens": 1024, "nframes": 24,
                             "frame_mode": "uniform_end", "meta": {"cond": cond, "question": q, "cuid": cuid,
                                                                   "in_paper_set": cuid in paper.get(q, {})}})
    dump("egotom", rows)


def egotempo():
    ann = json.load(open(f"{P}/egotempo_openQA.json"))["annotations"]
    rows = []
    for a in ann:
        cid = a["clip_id"][:-4] if a["clip_id"].endswith(".mp4") else a["clip_id"]
        parts = cid.split("_")
        clip = f"{'_'.join(parts[:-2])}_{float(parts[-2])}_{float(parts[-1])}.mp4"
        user = (f"These are frames from a video that I want to upload. Use the visual cues to answer the question: {a['question']}. "
                f"You need to answer the question in any case and not demand additional context information. "
                f"Note: All actions mentioned refer to the person recording the video.")
        rows.append({"id": a["question_id"], "bench": "egotempo", "video": f"egotempo/clips/{clip}", "user": user,
                     "question": a["question"], "max_new_tokens": 1024, "nframes": 32, "frame_mode": "uniform",
                     "meta": {"question_type": a["question_type"]}})
    dump("egotempo", rows)


def egosound():
    rows = []
    for f in ("ego4d", "egoblind"):
        for a in json.load(open(f"{P}/egosound_{f}.json")):
            v = a["video_path"]
            rows.append({"id": a["question_id"] if f == "ego4d" else a["question_id"], "bench": "egosound",
                         "video": f"egosound/{v}", "audio": f"egosound/{v.replace('videos', 'audios').replace('.mp4', '.wav')}",
                         "user": a["question"], "question": a["question"], "max_new_tokens": 1024, "nframes": 32,
                         "frame_mode": "uniform", "meta": {"subset": f, "question_type": a["question_type"]}})
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "EgoSound question_id collision across subsets"
    dump("egosound", rows)


if __name__ == "__main__":
    egotom()
    egotempo()
    egosound()
