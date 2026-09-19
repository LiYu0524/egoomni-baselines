"""Prompt templates and the MCQ letter parser. Every template change here changes the protocol — bump PROMPT_VERSION."""
import re

PROMPT_VERSION = "v1"
OPEN_EN = "{q}\nAnswer briefly in one or two sentences."
OPEN_ZH = "{q}\n请用一两句话简要回答。"
MCQ_EN = "{q}\n\nOptions:\n{opts}\nAnswer with the option's letter from the given choices directly."
MCQ_ZH = "{q}\n\n选项：\n{opts}\n请直接回答正确选项的字母。"


def format_options(options: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def format_question(question: str, fmt: str, lang: str, options: dict | None = None) -> str:
    if fmt == "mcq":
        assert options, "mcq without options"
        tpl = MCQ_ZH if lang == "zh" else MCQ_EN
        return tpl.format(q=question.strip(), opts=format_options(options))
    tpl = OPEN_ZH if lang == "zh" else OPEN_EN
    return tpl.format(q=question.strip())


_LEAD = re.compile(r"^\W*\(?([A-Ha-h])\)?(?:[\.\):：,，]|\s|$)")
_KEY = re.compile(r"(?:answer|option|choice|选项|答案)\s*(?:is|:|：|is:|为|是)?\s*\(?([A-Ha-h])\)?(?:[\.\):：,，]|\s|$)", re.I)
_ANY = re.compile(r"(?<![A-Za-z])\(?([A-H])\)?(?:[\.\):：]|\s|$)")
_TRAIL = re.compile(r"(?<![A-Za-z])([A-Ha-h])[\.\s]*$")


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", s.lower()).strip()


def parse_mcq(pred: str | None, options: dict) -> str | None:
    """Return the chosen option letter or None when the prediction is not attributable to exactly one option."""
    if not pred:
        return None
    letters = {k.upper() for k in options}
    p = pred.strip()
    standalone = {m.group(1) for m in _ANY.finditer(p) if m.group(1) in letters}   # every letter used as an option token
    m = _LEAD.match(p)
    if m and m.group(1).upper() in letters and standalone <= {m.group(1).upper()}:
        return m.group(1).upper()
    m = _KEY.search(p)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    np_ = _norm(p)
    hits = [k.upper() for k, v in options.items() if _norm(v) and _norm(v) in np_]
    if len(hits) == 1:
        return hits[0]
    if len(standalone) == 1:
        return standalone.pop()
    m = _TRAIL.search(p)
    if m and m.group(1).upper() in letters and not standalone:
        return m.group(1).upper()
    return None
