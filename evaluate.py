"""Quantitative checks you can quote in the report.

  python evaluate.py asr --audio meeting.wav --reference meeting_reference.txt   # word error rate
  python evaluate.py doc [--cases eval/cases.json] [--no-verify]                  # decisions / tasks quality
"""
import argparse
import json
import re
from difflib import SequenceMatcher

from meeting_ai.schema import UNSPEC


def _tok(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def wer(reference, hypothesis):
    r, h = _tok(reference), _tok(hypothesis)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / max(1, len(r))


def _match(text, golds, thr=0.5):
    best, bi = 0.0, None
    for i, g in enumerate(golds):
        s = SequenceMatcher(None, _tok(text), _tok(g)).ratio()
        if s > best:
            best, bi = s, i
    return bi if best >= thr else None


def _prf(pred, gold, key):
    used, tp = set(), 0
    for p in pred:
        i = _match(key(p), [key(g) for g in gold])
        if i is not None and i not in used:
            used.add(i)
            tp += 1
    return tp, len(pred), len(gold)


def run_doc(cases_path, verify):
    from meeting_ai import config as cfg
    from meeting_ai.document import generate_record
    from meeting_ai.models import LLM
    llm = LLM(cfg.DOC_MODEL, 0, cfg.LOAD_4BIT)
    llm.load()
    tot = {"d": [0, 0, 0], "a": [0, 0, 0]}
    invented_owner = invented_deadline = unspecified_gold = 0
    for case in json.load(open(cases_path)):
        rec, _ = generate_record(llm, case["transcript"], verify)
        for k, pred, gold, key in (("d", rec["decisions"], case["decisions"], lambda x: x["decision"] if isinstance(x, dict) else x),
                                   ("a", rec["action_items"], case["actions"], lambda x: x["task"])):
            for t, v in zip(range(3), _prf(pred, gold, key)):
                tot[k][t] += v
        for p in rec["action_items"]:
            i = _match(p["task"], [g["task"] for g in case["actions"]])
            if i is not None:
                g = case["actions"][i]
                unspecified_gold += 1
                invented_owner += g["owner"] == UNSPEC and p["owner"] != UNSPEC
                invented_deadline += g["deadline"] == UNSPEC and p["deadline"] != UNSPEC
        print(f"{case['name']:<55} decisions={len(rec['decisions'])} (gold {len(case['decisions'])})  actions={len(rec['action_items'])} (gold {len(case['actions'])})")
    for name, (tp, npred, ngold) in (("decisions", tot["d"]), ("action items", tot["a"])):
        print(f"{name:<13} precision={tp / max(1, npred):.2f}  recall={tp / max(1, ngold):.2f}  (tp={tp}, predicted={npred}, gold={ngold})")
    print(f"invented owners: {invented_owner}, invented deadlines: {invented_deadline} (out of {unspecified_gold} matched tasks)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("asr")
    a.add_argument("--audio", required=True)
    a.add_argument("--reference", required=True)
    d = sub.add_parser("doc")
    d.add_argument("--cases", default="eval/cases.json")
    d.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    if args.mode == "asr":
        from meeting_ai import config as cfg
        from meeting_ai.audio import prepare_audio
        from meeting_ai.models import Transcriber
        wav, _ = prepare_audio(args.audio, cfg.WORK_DIR)
        asr = Transcriber(cfg.ASR_MODEL, 0, cfg.ASR_BATCH_SIZE)
        asr.load()
        hyp = asr.transcribe(wav)
        print(f"WER = {wer(open(args.reference).read(), hyp):.3f}\n\nHYPOTHESIS:\n{hyp}")
    else:
        run_doc(args.cases, not args.no_verify)
