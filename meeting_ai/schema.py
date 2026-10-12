"""Single source of truth for the meeting-record schema, plus tolerant normalisation.

Final record:
{
  "summary": str,
  "minutes":      [{"topic": str, "discussion": str}],
  "decisions":    [{"decision": str, "evidence": str}],
  "action_items": [{"task": str, "owner": str, "deadline": str, "evidence": str}]
}
Models drift in key names, so every model output goes through the normalisers below.
Nothing is silently replaced by "Unspecified" except missing owners/deadlines; every repair is logged.
"""
import json
import re

UNSPEC = "Unspecified"

_NONE_WORDS = {"", "none", "n/a", "na", "null", "nil", "unknown", "unspecified", "not specified",
               "not stated", "not mentioned", "tbd", "-", "—"}

TASK_KEYS = ("task", "action", "action_item", "description", "item", "title", "todo", "work")
OWNER_KEYS = ("owner", "assignee", "assigned_to", "responsible", "who", "person")
DEADLINE_KEYS = ("deadline", "due", "due_date", "due_by", "by", "when", "date")
TOPIC_KEYS = ("topic", "title", "heading", "subject", "name")
DISCUSSION_KEYS = ("discussion", "summary", "details", "description", "notes", "text", "content", "points")
DECISION_KEYS = ("decision", "text", "description", "statement", "item", "content")
EVIDENCE_KEYS = ("evidence", "quote", "source_quote", "support")


def is_none_text(v):
    if v is None:
        return True
    if isinstance(v, (list, tuple, dict)):
        return len(v) == 0
    return str(v).strip().strip(".").lower() in _NONE_WORDS


def as_text(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(as_text(x) for x in v if not is_none_text(x))
    if isinstance(v, dict):
        return "; ".join(f"{k}: {as_text(x)}" for k, x in v.items() if not is_none_text(x))
    return " ".join(str(v).split())


def pick(d, keys):
    for k in keys:
        if k in d and not is_none_text(d[k]):
            return as_text(d[k])
    return ""


def as_list(v):
    if is_none_text(v):
        return []
    if isinstance(v, (str, dict)):
        return [v]
    if isinstance(v, (list, tuple)):
        return list(v)
    return []


def norm_unspecified(v):
    return UNSPEC if is_none_text(v) else as_text(v)


# ------------------------------------------------------------------ JSON extraction
def extract_json(text):
    """Return the first JSON object found in a model reply (tolerates fences and stray prose)."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", t):
        try:
            obj, _ = dec.raw_decode(t[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ------------------------------------------------------------------ normalisers
def _split_topic(text, n):
    m = re.match(r"^([^:]{3,60}):\s+(.+)$", text)
    if m and len(m.group(1).split()) <= 8:
        return m.group(1).strip(), m.group(2).strip()
    return f"Discussion point {n}", text


def normalize_minutes(value, repairs=None):
    repairs = [] if repairs is None else repairs
    out = []
    for n, m in enumerate(as_list(value), 1):
        if isinstance(m, str):
            text = as_text(m)
            if is_none_text(text):
                continue
            topic, disc = _split_topic(text, n)
            repairs.append("minute returned as plain string; topic split or derived")
            out.append({"topic": topic, "discussion": disc})
        elif isinstance(m, dict):
            topic, disc = pick(m, TOPIC_KEYS), pick(m, DISCUSSION_KEYS)
            if not topic and not disc:
                repairs.append("dropped empty minute")
                continue
            out.append({"topic": topic or f"Discussion point {n}", "discussion": disc})
    return out


def normalize_decisions(value, repairs=None):
    repairs = [] if repairs is None else repairs
    out = []
    for d in as_list(value):
        if isinstance(d, str):
            text, ev = as_text(d), ""
        elif isinstance(d, dict):
            text, ev = pick(d, DECISION_KEYS), pick(d, EVIDENCE_KEYS)
            if not text:
                repairs.append("dropped decision without text")
                continue
        else:
            continue
        if is_none_text(text):
            continue
        out.append({"decision": text, "evidence": ev})
    return out


def normalize_actions(value, repairs=None):
    repairs = [] if repairs is None else repairs
    out = []
    for a in as_list(value):
        if isinstance(a, str):
            task = as_text(a)
            if is_none_text(task):
                continue
            repairs.append("action item returned as plain string")
            out.append({"task": task, "owner": UNSPEC, "deadline": UNSPEC, "evidence": ""})
        elif isinstance(a, dict):
            task = pick(a, TASK_KEYS)
            if not task:
                repairs.append("dropped action item without task text")
                continue
            out.append({
                "task": task,
                "owner": norm_unspecified(pick(a, OWNER_KEYS)),
                "deadline": norm_unspecified(pick(a, DEADLINE_KEYS)),
                "evidence": pick(a, EVIDENCE_KEYS),
            })
    return out


def validate_meeting_record(record, repairs=None):
    """Normalise a (possibly messy) record into the canonical schema. Idempotent."""
    repairs = [] if repairs is None else repairs
    if not isinstance(record, dict):
        repairs.append("model output was not a JSON object")
        record = {}
    summary = record.get("summary")
    return {
        "summary": "" if is_none_text(summary) else as_text(summary),
        "minutes": normalize_minutes(record.get("minutes"), repairs),
        "decisions": normalize_decisions(record.get("decisions"), repairs),
        "action_items": normalize_actions(record.get("action_items"), repairs),
    }
