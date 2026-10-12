"""LLM #2 - structured meeting documentation.

Instead of one giant prompt, extraction is staged (minutes, decisions, action items, summary) and runs over
overlapping windows so long meetings are fully covered. Hallucination control is enforced in code:

  1. every decision / action item must carry a verbatim evidence quote that is found in the transcript;
  2. owners and deadlines must literally occur in the transcript, otherwise they become "Unspecified";
  3. an optional verifier pass asks the model whether the quote really shows agreement / commitment
     (a proposal is not a decision, an unassigned "someone should" is not an action item);
  4. rejected candidates are kept in an audit list instead of disappearing silently.
"""
from difflib import SequenceMatcher

from . import config as cfg
from .grounding import (Indexed, context_window, evidence_text, find_quote, norm_token, value_supported)
from .schema import (UNSPEC, extract_json, normalize_actions, normalize_decisions, normalize_minutes,
                     validate_meeting_record)
from .textutils import chunk_sentences

SYSTEM = "You are a precise meeting-documentation assistant. Reply with a single JSON object and nothing else."

MINUTES_PROMPT = """TASK: MINUTES
Read the meeting transcript below and write organised minutes.

Return JSON: {"minutes": [{"topic": "<short topic title>", "discussion": "<1-3 factual sentences>"}]}

Rules:
- Cover only topics that were actually discussed. Keep each discussion factual and brief.
- Describe proposals as proposals ("it was suggested that...") and disagreements as disagreements. Never describe a suggestion as agreed.
- Do not add names, numbers or dates that are not in the transcript.
- If the excerpt has no substantive discussion, return {"minutes": []}.

=== TRANSCRIPT ===
<<TRANSCRIPT>>"""

DECISIONS_PROMPT = """TASK: DECISIONS
Find the decisions in the meeting transcript below. A decision exists ONLY if the participants clearly agreed on,
approved, confirmed or finalised something ("we decided", "let's go with X" followed by agreement, "it's settled").
These are NOT decisions: suggestions ("we should", "maybe", "I think"), questions, preferences, ideas to explore,
things postponed ("let's talk next week"), and unresolved disagreements.

For each decision give a verbatim quote from the transcript as evidence (copy the exact words, at most 40 words).

Return JSON: {"decisions": [{"decision": "<one sentence>", "evidence": "<verbatim quote>"}]}
If there is no decision, return {"decisions": []}.

EXAMPLE (not part of the meeting)
Transcript: I think we should maybe switch to Postgres. Could be good, but the migration worries me. Okay, let's go with Postgres then. Everyone agreed. Should we also move the cache to Redis? Let's talk about that next week.
Output: {"decisions": [{"decision": "The team agreed to switch the database to Postgres.", "evidence": "Okay, let's go with Postgres then. Everyone agreed."}]}
(The Redis idea is a question that was postponed, so it is not listed.)

=== TRANSCRIPT ===
<<TRANSCRIPT>>"""

ACTIONS_PROMPT = """TASK: ACTIONS
Find the action items in the meeting transcript below. An action item exists ONLY if someone is asked to do something
and accepts, volunteers for it, or it is clearly assigned or committed to. "Someone should...", "we could...", "maybe
we need to..." are NOT action items.

- task: the work to be done.
- owner: a person's name ONLY if the transcript names who is responsible. Otherwise "Unspecified". Never guess.
- deadline: ONLY if a date or time limit is stated for that task. Otherwise "Unspecified". Never guess.
- evidence: a verbatim quote from the transcript (exact words, at most 40 words).

Return JSON: {"action_items": [{"task": "...", "owner": "...", "deadline": "...", "evidence": "..."}]}
If there are none, return {"action_items": []}.

EXAMPLE (not part of the meeting)
Transcript: Tom, can you write the migration script? Sure, I'll have it done by Friday. Someone should also look at the monitoring dashboards at some point. Yeah, maybe.
Output: {"action_items": [{"task": "Write the migration script", "owner": "Tom", "deadline": "Friday", "evidence": "Tom, can you write the migration script? Sure, I'll have it done by Friday."}]}
(The dashboards remark is not listed: nobody committed to it.)

=== TRANSCRIPT ===
<<TRANSCRIPT>>"""

SUMMARY_PROMPT = """TASK: SUMMARY
Write a concise summary (2-4 sentences) of the meeting using ONLY the information below. Do not add anything else.
Do not describe a proposal as a decision. Mention a decision only if it appears in the DECISIONS list.

Return JSON: {"summary": "..."}

MINUTES:
<<MINUTES>>

DECISIONS:
<<DECISIONS>>"""

VERIFY_DECISION_PROMPT = """TASK: VERIFY_DECISION
Meeting excerpt:
<<CONTEXT>>

Candidate decision: <<ITEM>>

Did the participants clearly AGREE on, approve, confirm or finalise this in the excerpt?
A suggestion, question, preference, idea, postponed topic or "we should/could" is NOT a decision.
Return JSON: {"verdict": "agreed"} or {"verdict": "proposed"} or {"verdict": "unclear"}"""

VERIFY_ACTION_PROMPT = """TASK: VERIFY_ACTION
Meeting excerpt:
<<CONTEXT>>

Candidate action item: <<ITEM>>

In the excerpt, was this work assigned to someone, volunteered for, or clearly committed to?
Vague remarks like "someone should", "we could", "maybe" are NOT commitments.
Return JSON: {"verdict": "committed"} or {"verdict": "discussed_only"}"""


# ------------------------------------------------------------------ helpers
def _fill(template, **kw):
    for k, v in kw.items():
        template = template.replace(f"<<{k}>>", v)
    return template


def ask_json(llm, prompt, max_new_tokens, warnings, label):
    """Call the model, retrying with a stricter reminder if the reply is not valid JSON."""
    user = prompt
    for _ in range(cfg.JSON_RETRIES + 1):
        data = extract_json(llm.generate(SYSTEM, user, max_new_tokens))
        if data is not None:
            return data
        user = prompt + "\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON object, no other text."
    warnings.append(f"{label}: no valid JSON after {cfg.JSON_RETRIES + 1} attempts")
    return {}


def _verdict(llm, template, item, context, warnings, label):
    data = ask_json(llm, _fill(template, CONTEXT=context, ITEM=item), 40, warnings, label)
    return str(data.get("verdict", "unclear")).strip().lower()


def _similar(a, b, thr):
    ta = [t for t in (norm_token(w) for w in a.split()) if t]
    tb = [t for t in (norm_token(w) for w in b.split()) if t]
    return bool(ta and tb) and SequenceMatcher(None, ta, tb, autojunk=False).ratio() >= thr


def _specified(a):
    return (a["owner"] != UNSPEC) + (a["deadline"] != UNSPEC)


def _dedupe_decisions(items):
    out = []
    for d in items:
        if not any(_similar(d["decision"], e["decision"], 0.75) for e in out):
            out.append(d)
    return out


def _dedupe_actions(items):
    out = []
    for a in items:
        for i, e in enumerate(out):
            if _similar(a["task"], e["task"], 0.75):
                if _specified(a) > _specified(e):
                    out[i] = a
                break
        else:
            out.append(a)
    return out


def _merge_minutes(minutes):
    merged = []
    for m in minutes:
        for e in merged:
            if _similar(e["topic"], m["topic"], 0.85):
                if m["discussion"] and not _similar(e["discussion"], m["discussion"], 0.7):
                    e["discussion"] = (e["discussion"] + " " + m["discussion"]).strip()
                break
        else:
            merged.append(dict(m))
    return merged


# ------------------------------------------------------------------ candidate checks
def _check_decision(llm, d, idx, verify, warnings):
    hit = find_quote(d["evidence"], idx, cfg.EVIDENCE_THRESHOLD)
    if not hit:
        return None, "evidence quote not found in the transcript"
    _, s, e = hit
    d = {"decision": d["decision"], "evidence": evidence_text(idx, s, e)}
    if verify:
        v = _verdict(llm, VERIFY_DECISION_PROMPT, d["decision"], context_window(idx, s, e), warnings, "verify-decision")
        if v != "agreed":
            return None, f"verifier judged it '{v}', not an agreed decision"
    return d, ""


def _check_action(llm, a, idx, verify, warnings, repairs):
    hit = find_quote(a["evidence"], idx, cfg.EVIDENCE_THRESHOLD)
    if not hit:
        return None, "evidence quote not found in the transcript"
    _, s, e = hit
    a = dict(a, evidence=evidence_text(idx, s, e))
    if a["owner"] != UNSPEC and not value_supported(a["owner"], idx):
        repairs.append(f"owner '{a['owner']}' not found in transcript -> Unspecified")
        a["owner"] = UNSPEC
    if a["deadline"] != UNSPEC and not value_supported(a["deadline"], idx):
        repairs.append(f"deadline '{a['deadline']}' not found in transcript -> Unspecified")
        a["deadline"] = UNSPEC
    if verify:
        v = _verdict(llm, VERIFY_ACTION_PROMPT, a["task"], context_window(idx, s, e), warnings, "verify-action")
        if v != "committed":
            return None, f"verifier judged it '{v}', not a committed task"
    return a, ""


# ------------------------------------------------------------------ main entry points
def generate_record_iter(llm, refined, verify=None):
    """Generator: yields {'message'} progress items, then a final {'done': True, 'record', 'audit'}."""
    verify = cfg.VERIFY_WITH_LLM if verify is None else verify
    warnings, repairs, rejected = [], [], []
    minutes, decisions, actions = [], [], []
    windows = chunk_sentences(refined, cfg.DOC_WINDOW_WORDS, cfg.DOC_OVERLAP_WORDS) or [refined]
    n = len(windows)

    for k, w in enumerate(windows, 1):
        idx = Indexed(w)
        part = f" ({k}/{n})" if n > 1 else ""

        yield {"message": f"Writing minutes{part}…"}
        data = ask_json(llm, _fill(MINUTES_PROMPT, TRANSCRIPT=w), 1500, warnings, "minutes")
        minutes += normalize_minutes(data.get("minutes"), repairs)

        yield {"message": f"Finding decisions{part}…"}
        data = ask_json(llm, _fill(DECISIONS_PROMPT, TRANSCRIPT=w), 800, warnings, "decisions")
        for cand in normalize_decisions(data.get("decisions"), repairs):
            ok, why = _check_decision(llm, cand, idx, verify, warnings)
            if ok:
                decisions.append(ok)
            else:
                rejected.append({"type": "decision", "text": cand["decision"], "reason": why})

        yield {"message": f"Finding action items{part}…"}
        data = ask_json(llm, _fill(ACTIONS_PROMPT, TRANSCRIPT=w), 1000, warnings, "actions")
        for cand in normalize_actions(data.get("action_items"), repairs):
            ok, why = _check_action(llm, cand, idx, verify, warnings, repairs)
            if ok:
                actions.append(ok)
            else:
                rejected.append({"type": "action_item", "text": cand["task"], "reason": why})

    decisions, actions, minutes = _dedupe_decisions(decisions), _dedupe_actions(actions), _merge_minutes(minutes)

    yield {"message": "Writing summary…"}
    summary = ""
    if minutes or decisions:
        data = ask_json(llm, _fill(
            SUMMARY_PROMPT,
            MINUTES="\n".join(f"- {m['topic']}: {m['discussion']}" for m in minutes) or "(none)",
            DECISIONS="\n".join(f"- {d['decision']}" for d in decisions) or "(none)"), 400, warnings, "summary")
        summary = str(data.get("summary", "")).strip()
    if not summary:
        summary = "No substantive discussion could be identified in this recording." if not minutes else \
            "Topics discussed: " + "; ".join(m["topic"] for m in minutes) + "."

    record = validate_meeting_record({"summary": summary, "minutes": minutes,
                                      "decisions": decisions, "action_items": actions}, repairs)
    audit = {"windows": n, "verified_with_llm": verify, "rejected_candidates": rejected,
             "repairs": sorted(set(repairs))[:50], "warnings": warnings}
    yield {"done": True, "record": record, "audit": audit}


def generate_record(llm, refined, verify=None):
    last = None
    for last in generate_record_iter(llm, refined, verify):
        pass
    return last["record"], last["audit"]
