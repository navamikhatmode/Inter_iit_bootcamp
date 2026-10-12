"""LLM #1 - domain-aware transcript refinement, with programmatic guards.

Each chunk is refined with the tail of the previous (refined) chunk as read-only context. A refined chunk is
accepted only if it passes cheap checks (length, numbers, negations, similarity to the raw chunk); otherwise
the raw chunk is kept and the reason is recorded. So refinement can never silently summarise or flip meaning.
"""
import re
from collections import Counter
from difflib import SequenceMatcher

from . import config as cfg
from .textutils import chunk_sentences, wc

REFINE_SYSTEM = "You correct speech-to-text errors in meeting transcripts without changing their meaning."

REFINE_PROMPT = """You are refining ONE segment of a meeting transcript produced by automatic speech recognition.

Fix ONLY likely recognition errors, especially in:
- technical and scientific terms, engineering terms, acronyms, software and model names, units
- people's names and product names that are clearly misspelled (use the glossary and context below)
- obvious homophone mistakes that make a technical term nonsensical

Hard rules:
1. Keep the original wording, order and sentence structure. Do not rewrite, reorder, shorten or summarise.
2. Keep every number, date, unit and name exactly as heard unless it is clearly a recognition error of a glossary term.
3. Keep negations (not, no, never, n't, cannot) exactly as they are.
4. Keep questions, uncertainty ("maybe", "I think"), proposals, disagreements and commitments exactly as they are. Never turn a proposal into a decision.
5. Do not add owners, deadlines, decisions or any information that is not in the segment.
6. If you are not sure that a word is wrong, keep it.
7. Output ONLY the corrected segment text: no preface, no notes, no quotes.

Glossary of expected terms (may be empty): <<GLOSSARY>>

Previous text, for context only (do NOT output it): <<CONTEXT>>

SEGMENT TO CORRECT:
<<TEXT>>"""

_NEG_RE = re.compile(r"\b(?:not|no|never|none|nobody|nothing|neither|nor|cannot|without|\w+n't)\b", re.I)
_PREAMBLE_RE = re.compile(r"^\s*(?:sure|here(?: is|'s)|corrected|refined)[^\n]{0,80}:\s*\n+", re.I)


def build_prompt(chunk, context="", glossary=""):
    return (REFINE_PROMPT.replace("<<GLOSSARY>>", glossary.strip() or "none")
            .replace("<<CONTEXT>>", context.strip() or "none")
            .replace("<<TEXT>>", chunk))


def strip_preamble(text):
    t = (text or "").strip()
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t)
    t = _PREAMBLE_RE.sub("", t)
    if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1]
    return t.strip()


def count_negations(text):
    return len(_NEG_RE.findall(text.replace("’", "'")))


def numbers_preserved(raw, refined):
    need, have = Counter(re.findall(r"\d+", raw)), Counter(re.findall(r"\d+", refined))
    return all(have[k] >= v for k, v in need.items())


def check_refinement(raw, refined):
    """Return a list of problems; an empty list means the refined chunk is safe to use."""
    if not refined.strip():
        return ["empty output"]
    problems = []
    ratio = wc(refined) / max(1, wc(raw))
    if not (cfg.REFINE_MIN_RATIO <= ratio <= cfg.REFINE_MAX_RATIO):
        problems.append(f"length changed too much (ratio {ratio:.2f})")
    if not numbers_preserved(raw, refined):
        problems.append("a number was changed or removed")
    if count_negations(raw) != count_negations(refined):
        problems.append("negation count changed")
    sim = SequenceMatcher(None, raw.lower().split(), refined.lower().split(), autojunk=False).ratio()
    if sim < cfg.REFINE_MIN_SIMILARITY:
        problems.append(f"too different from the raw text (similarity {sim:.2f})")
    return problems


def refine_transcript_iter(llm, raw, glossary=""):
    """Generator: yields {'i','n','text','report','done'} after every chunk."""
    chunks = chunk_sentences(raw, cfg.REFINE_CHUNK_WORDS)
    out, report = [], []
    if not chunks:
        yield {"i": 0, "n": 0, "text": "", "report": [], "done": True}
        return
    for i, chunk in enumerate(chunks, 1):
        context = " ".join(" ".join(out).split()[-cfg.REFINE_CONTEXT_WORDS:])
        try:
            cand = strip_preamble(llm.generate(REFINE_SYSTEM, build_prompt(chunk, context, glossary),
                                               int(wc(chunk) * 2.2) + 64))
            problems = check_refinement(chunk, cand)
        except Exception as exc:  # noqa: BLE001 - recorded, raw text is kept
            cand, problems = "", [f"model error: {exc}"]
        fallback = bool(problems)
        out.append(chunk if fallback else cand)
        report.append({"chunk": i, "fallback": fallback, "problems": problems})
        yield {"i": i, "n": len(chunks), "text": " ".join(out), "report": report, "done": i == len(chunks)}


def refine_transcript(llm, raw, glossary=""):
    last = None
    for last in refine_transcript_iter(llm, raw, glossary):
        pass
    return last["text"], last["report"]
