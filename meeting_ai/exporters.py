"""Renderers and file writers. JSON / TXT / HTML all come from the same record, so they always agree."""
import datetime
import difflib
import html
import json
import os

from .schema import UNSPEC

CSS = """
:root{--bg:#ffffff;--bg2:#f8fafc;--bd:#e2e8f0;--tx:#0f172a;--mut:#64748b;--ac:#4f46e5;--ok:#10b981;--bad:#dc2626}
@media (prefers-color-scheme: dark){:root{--bg:#0f172a;--bg2:#1e293b;--bd:#334155;--tx:#e2e8f0;--mut:#94a3b8;--ac:#818cf8}}
.mi h2{font-size:1.1rem;margin:22px 0 10px}
.mi-card{background:var(--bg2);border:1px solid var(--bd);border-radius:12px;padding:14px 18px;margin-bottom:10px;color:var(--tx);line-height:1.6}
.mi-card .q{display:block;margin-top:6px;font-size:.85rem;color:var(--mut);font-style:italic}
.mi-empty{padding:22px;text-align:center;border:1.5px dashed var(--bd);border-radius:12px;color:var(--mut)}
.mi-table{width:100%;border-collapse:collapse;color:var(--tx)}
.mi-table th{text-align:left;font-size:.75rem;text-transform:uppercase;color:var(--mut);padding:8px 10px;border-bottom:1px solid var(--bd)}
.mi-table td{padding:10px;border-bottom:1px solid var(--bd);vertical-align:top}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;background:rgba(79,70,229,.15);color:var(--ac);font-size:.85rem}
.badge.mut{background:rgba(148,163,184,.2);color:var(--mut)}
.diff{white-space:pre-wrap;line-height:1.8;color:var(--tx)}
.diff del{background:rgba(220,38,38,.18);text-decoration:line-through}
.diff ins{background:rgba(16,185,129,.22);text-decoration:none}
.steps{display:flex;gap:8px;margin:4px 0 10px}
.step{flex:1;text-align:center;padding:8px 4px;border-radius:10px;background:var(--bg2);border:1px solid var(--bd);color:var(--mut);font-size:.85rem}
.step.done{color:var(--ok);border-color:var(--ok)}
.step.active{color:var(--ac);border-color:var(--ac);font-weight:700}
.step.err{color:var(--bad);border-color:var(--bad);font-weight:700}
.msg{color:var(--mut);font-size:.9rem}.msg.err{color:var(--bad);font-weight:600}
"""


def _e(x):
    return html.escape("" if x is None else str(x))


def _badge(v):
    v = v or UNSPEC
    return f'<span class="badge{" mut" if v == UNSPEC else ""}">{_e(v)}</span>'


def _empty(msg):
    return f'<div class="mi-empty">{_e(msg)}</div>'


def render_record_html(record):
    p = ["<div class='mi'><h2>Summary</h2>",
         f"<div class='mi-card'>{_e(record['summary'])}</div>" if record["summary"] else _empty("No summary generated."),
         "<h2>Meeting minutes</h2>"]
    p += [f"<div class='mi-card'><b>{i}. {_e(m['topic'])}</b><br>{_e(m['discussion'])}</div>"
          for i, m in enumerate(record["minutes"], 1)] or [_empty("No distinct discussion topics were identified.")]
    p.append("<h2>Key decisions</h2>")
    p += [f"<div class='mi-card'>✓ {_e(d['decision'])}" + (f"<span class='q'>“{_e(d['evidence'])}”</span>" if d["evidence"] else "") + "</div>"
          for d in record["decisions"]] or [_empty("No explicit decisions were identified in this meeting.")]
    p.append("<h2>Action items</h2>")
    if record["action_items"]:
        rows = "".join(f"<tr><td>{i}</td><td>{_e(a['task'])}" + (f"<span class='q' style='display:block;font-size:.8rem;font-style:italic;color:var(--mut)'>“{_e(a['evidence'])}”</span>" if a["evidence"] else "") +
                       f"</td><td>{_badge(a['owner'])}</td><td>{_badge(a['deadline'])}</td></tr>"
                       for i, a in enumerate(record["action_items"], 1))
        p.append("<table class='mi-table'><thead><tr><th>#</th><th>Task</th><th>Owner</th><th>Deadline</th></tr></thead>"
                 f"<tbody>{rows}</tbody></table>")
    else:
        p.append(_empty("No explicit action items were identified in this meeting."))
    p.append("</div>")
    return "".join(p)


def diff_html(raw, refined):
    a, b = raw.split(), refined.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(_e(" ".join(a[i1:i2])))
            continue
        if i2 > i1:
            out.append(f"<del>{_e(' '.join(a[i1:i2]))}</del>")
        if j2 > j1:
            out.append(f"<ins>{_e(' '.join(b[j1:j2]))}</ins>")
    return "<div class='mi'><div class='diff'>" + " ".join(out) + "</div></div>"


def to_json(record, audit):
    return json.dumps({**record, "audit": audit}, indent=2, ensure_ascii=False)


def to_text(record):
    L = ["MEETING RECORD", "=" * 60, "", "SUMMARY", "-" * 60, record["summary"] or "(none)", "",
         "MEETING MINUTES", "-" * 60]
    for i, m in enumerate(record["minutes"], 1):
        L += [f"{i}. {m['topic']}", f"   {m['discussion']}"]
    if not record["minutes"]:
        L.append("No distinct discussion topics were identified.")
    L += ["", "KEY DECISIONS", "-" * 60]
    for i, d in enumerate(record["decisions"], 1):
        L.append(f"{i}. {d['decision']}")
        if d["evidence"]:
            L.append(f"   Evidence: \"{d['evidence']}\"")
    if not record["decisions"]:
        L.append("No explicit decisions were identified.")
    L += ["", "ACTION ITEMS", "-" * 60]
    for i, a in enumerate(record["action_items"], 1):
        L += [f"{i}. {a['task']}", f"   Owner: {a['owner']}", f"   Deadline: {a['deadline']}"]
        if a["evidence"]:
            L.append(f"   Evidence: \"{a['evidence']}\"")
    if not record["action_items"]:
        L.append("No explicit action items were identified.")
    return "\n".join(L) + "\n"


def to_html_report(record, raw, refined, audit, meta):
    rej = audit.get("rejected_candidates", [])
    rej_html = "".join(f"<li><b>{_e(r['type'])}</b>: {_e(r['text'])} <i>({_e(r['reason'])})</i></li>" for r in rej) or "<li>none</li>"
    stamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    return (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Meeting Record</title><style>body{{font-family:system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--tx);max-width:900px;margin:0 auto;padding:24px}}"
            f"pre{{white-space:pre-wrap}}{CSS}</style></head><body><h1>Meeting Record</h1>"
            f"<p class='msg'>Generated {stamp} · audio {_e(meta.get('duration_text', '-'))} · {_e(meta.get('words', '-'))} words · "
            f"ASR {_e(meta.get('asr_model'))} · refiner {_e(meta.get('refiner_model'))} · documentation {_e(meta.get('doc_model'))}</p>"
            f"{render_record_html(record)}<h2>Raw → refined (changes highlighted)</h2>{diff_html(raw, refined)}"
            f"<details><summary>Refined transcript</summary><pre>{_e(refined)}</pre></details>"
            f"<details><summary>Raw transcript</summary><pre>{_e(raw)}</pre></details>"
            f"<details><summary>Audit: candidates rejected by the grounding checks</summary><ul>{rej_html}</ul></details></body></html>")


def write_outputs(out_dir, record, raw, refined, audit, meta):
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "json": ("meeting_record.json", to_json(record, audit)),
        "txt": ("meeting_record.txt", to_text(record)),
        "html": ("meeting_report.html", to_html_report(record, raw, refined, audit, meta)),
        "raw": ("raw_transcript.txt", raw + "\n"),
        "refined": ("refined_transcript.txt", refined + "\n"),
    }
    paths = {}
    for key, (name, content) in files.items():
        paths[key] = os.path.join(out_dir, name)
        with open(paths[key], "w", encoding="utf-8") as f:
            f.write(content)
    return paths
