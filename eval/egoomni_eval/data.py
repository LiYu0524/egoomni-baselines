"""qa.json → flat list of answer Requests (one per turn), for the 'gold' and 'self' multi-turn protocols."""
import json
import os
import re
from dataclasses import dataclass, field, asdict

from . import QA_PATH, CLIPS_META, DATA_DIR
from .prompts import format_question

CJK = re.compile(r"[一-鿿]")
ASSIST_HEADER = "<|im_start|>assistant\n"


def item_id(it: dict) -> str:
    return it.get("qa_id") or it.get("sample_id")


def lang_of(text: str) -> str:
    return "zh" if CJK.search(text or "") else "en"


def turn_key(iid: str, turn_idx: int) -> str:
    return f"{iid}#r{turn_idx}"


@dataclass
class Request:
    key: str
    item_id: str
    turn_idx: int          # 1-based
    n_turns: int
    protocol: str          # gold | self
    clip: str              # absolute path
    clip_rel: str
    has_audio: bool
    duration: float
    lang: str
    fmt: str               # open | mcq
    question: str          # raw question of this turn
    messages: list         # [{role, content}], last is this turn's user message (already templated)
    gold: str
    options: dict | None
    correct_options: list
    meta: dict = field(default_factory=dict)
    dummy: bool = False    # padding row for lockstep (ZeRO-3) workers; never saved

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("clip")
        d.pop("dummy")
        return d


def load_items(path: str = QA_PATH) -> list[dict]:
    items = json.load(open(path))
    ids = [item_id(it) for it in items]
    assert len(set(ids)) == len(ids), "item ids not unique"
    return items


def load_clips_meta(path: str = CLIPS_META) -> dict:
    return json.load(open(path))


def _turns(it: dict) -> list[dict]:
    """Unified per-turn view: [{question, answer, fmt, options, correct_options, meta}]."""
    if it["source_kind"] == "multi_turn":
        out = []
        ot = {t["round_index"]: t for t in (it.get("original_turns") or [])}
        for t in it["qa"]:
            o = ot.get(t["turn_index"], {})
            out.append(dict(question=t["question"], answer=t["answer"], fmt=(o.get("question_format") or "open"),
                            options=None, correct_options=[],
                            meta=dict(target_component=o.get("target_component"), depends_on_rounds=o.get("depends_on_rounds"),
                                      turn_modalities=o.get("modalities"), standalone_question=o.get("standalone_question"))))
        return out
    fmt = it.get("original_question_format") or "open"
    return [dict(question=it["original_question"], answer=it["original_answer"], fmt=fmt,
                 options=it.get("options"), correct_options=it.get("correct_options") or [], meta={})]


def _item_meta(it: dict) -> dict:
    return dict(category=it.get("category"), subcategory=it.get("subcategory"), track=it.get("benchmark_track"),
                source_kind=it.get("source_kind"), min_modalities=it.get("minimum_modalities"), modalities=it.get("modalities"),
                round_count=it.get("round_count"), video_id=it.get("video_id"))


def build_requests(items: list[dict], protocol: str, clips_meta: dict, self_answers: dict | None = None,
                   round_idx: int | None = None, subset_ids: set | None = None) -> list[Request]:
    """gold: every single-turn item + every multi-turn round with dataset answers as history.
       self: multi-turn rounds == round_idx only, history = self_answers[turn_key] (model's own earlier answers)."""
    assert protocol in ("gold", "self")
    if protocol == "self":
        assert round_idx is not None and self_answers is not None
    reqs = []
    for it in items:
        iid = item_id(it)
        if subset_ids is not None and iid not in subset_ids:
            continue
        turns = _turns(it)
        multi = it["source_kind"] == "multi_turn"
        if protocol == "self" and not multi:
            continue
        rel = it["clip_path"]
        cm = clips_meta.get(rel)
        if cm is None:
            raise KeyError(f"clip not in clips_meta: {rel}")
        base_meta = _item_meta(it)
        for k, t in enumerate(turns, start=1):
            if protocol == "self" and k != round_idx:
                continue
            lang = lang_of(t["question"])
            msgs, missing = [], []
            for j in range(1, k):
                pj = turns[j - 1]
                msgs.append({"role": "user", "content": format_question(pj["question"], pj["fmt"], lang_of(pj["question"]), pj["options"])})
                if protocol == "gold":
                    hist = pj["answer"]
                else:
                    hist = (self_answers or {}).get(turn_key(iid, j))
                    if hist is None:
                        missing.append(j)
                        hist = ""
                msgs.append({"role": "assistant", "content": hist})
            msgs.append({"role": "user", "content": format_question(t["question"], t["fmt"], lang, t["options"])})
            meta = dict(base_meta, **t["meta"])
            if missing:
                meta["history_missing_rounds"] = missing
            reqs.append(Request(
                key=turn_key(iid, k), item_id=iid, turn_idx=k, n_turns=len(turns), protocol=protocol,
                clip=os.path.join(DATA_DIR, rel), clip_rel=rel, has_audio=bool(cm["has_audio"]), duration=float(cm["duration"]),
                lang=lang, fmt=t["fmt"], question=t["question"], messages=msgs, gold=t["answer"],
                options=t["options"], correct_options=t["correct_options"], meta=meta))
    return reqs


def read_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # truncated last line from a killed worker
    return rows


def read_pred_dir(d: str) -> list[dict]:
    """All rows under a preds dir (recursively: shards and rounds)."""
    rows = []
    for dp, _, fs in os.walk(d):
        for fn in sorted(fs):
            if fn.endswith(".jsonl"):
                rows.extend(read_jsonl(os.path.join(dp, fn)))
    return rows


def load_subset_ids(path: str | None) -> set | None:
    if not path:
        return None
    d = json.load(open(path))
    return set(d["item_ids"] if isinstance(d, dict) else d)


# ---------------------------------------------------------------------------------------------
# Item-level plan: decode each clip once, answer all its turns under both protocols in one worker.
# ---------------------------------------------------------------------------------------------
@dataclass
class ItemPlan:
    item_id: str
    clip: str
    clip_rel: str
    has_audio: bool
    duration: float
    multi: bool
    turns: list            # [{turn_idx, question, fmt, lang, options, correct_options, gold, meta}]
    meta: dict

    @property
    def n_turns(self):
        return len(self.turns)

    def n_generations(self, protocols):
        n = 0
        if "gold" in protocols:
            n += self.n_turns
        if "self" in protocols and self.multi:
            n += self.n_turns - (1 if "gold" in protocols else 0)   # self round 1 == gold round 1
        return n


def build_item_plans(items: list[dict], clips_meta: dict, subset_ids: set | None = None) -> list[ItemPlan]:
    plans = []
    for it in items:
        iid = item_id(it)
        if subset_ids is not None and iid not in subset_ids:
            continue
        rel = it["clip_path"]
        cm = clips_meta[rel]
        turns = []
        for k, t in enumerate(_turns(it), start=1):
            turns.append(dict(turn_idx=k, question=t["question"], fmt=t["fmt"], lang=lang_of(t["question"]), options=t["options"],
                              correct_options=t["correct_options"], gold=t["answer"], meta=t["meta"]))
        plans.append(ItemPlan(item_id=iid, clip=os.path.join(DATA_DIR, rel), clip_rel=rel, has_audio=bool(cm["has_audio"]),
                              duration=float(cm["duration"]), multi=it["source_kind"] == "multi_turn", turns=turns, meta=_item_meta(it)))
    return plans


def messages_for(plan: ItemPlan, k: int, history: list[str]) -> list[dict]:
    """Chat messages for turn k (1-based) given the assistant answers used for turns 1..k-1."""
    assert len(history) == k - 1
    msgs = []
    for j in range(1, k):
        pj = plan.turns[j - 1]
        msgs.append({"role": "user", "content": format_question(pj["question"], pj["fmt"], pj["lang"], pj["options"])})
        msgs.append({"role": "assistant", "content": history[j - 1] if history[j - 1] is not None else ""})
    t = plan.turns[k - 1]
    msgs.append({"role": "user", "content": format_question(t["question"], t["fmt"], t["lang"], t["options"])})
    return msgs


def request_for(plan: ItemPlan, k: int, protocol: str, history: list[str]) -> Request:
    t = plan.turns[k - 1]
    meta = dict(plan.meta, **t["meta"])
    if any(h is None for h in history):
        meta["history_missing_rounds"] = [j + 1 for j, h in enumerate(history) if h is None]
    return Request(key=turn_key(plan.item_id, k), item_id=plan.item_id, turn_idx=k, n_turns=plan.n_turns, protocol=protocol,
                   clip=plan.clip, clip_rel=plan.clip_rel, has_audio=plan.has_audio, duration=plan.duration, lang=t["lang"], fmt=t["fmt"],
                   question=t["question"], messages=messages_for(plan, k, history), gold=t["gold"], options=t["options"],
                   correct_options=t["correct_options"], meta=meta)
