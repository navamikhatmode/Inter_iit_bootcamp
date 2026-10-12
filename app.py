"""Gradio web app. Run:  python app.py   (SHARE=1 python app.py for a public gradio.live link)"""
import logging
import os

import gradio as gr

from meeting_ai import config as cfg
from meeting_ai.errors import PipelineError
from meeting_ai.exporters import CSS, render_record_html
from meeting_ai.pipeline import STAGES, MeetingPipeline

log = logging.getLogger("meeting_ai")


def status_html(stage, message, error=False):
    cells = []
    for i, name in enumerate(STAGES):
        cls = "done" if (stage >= 4 or i < stage) else ("err" if (error and i == stage) else ("active" if i == stage else ""))
        mark = "✓" if cls == "done" else ("✕" if cls == "err" else str(i + 1))
        cells.append(f"<div class='step {cls}'>{mark} {name}</div>")
    return f"<div class='mi'><div class='steps'>{''.join(cells)}</div><div class='msg{' err' if error else ''}'>{message}</div></div>"


def empty_note(text):
    return f"<div class='mi'><div class='mi-empty'>{text}</div></div>"


def build_app(pipe):
    def run(audio_path, hint):
        raw = refined = ""
        diff = empty_note("The raw-vs-refined comparison will appear here.")
        record = empty_note("The meeting record will appear here.")
        files = {}
        stage = 0

        def out(status):
            return (status, raw, refined, diff, record, files.get("json"), files.get("txt"),
                    files.get("html"), files.get("raw"), files.get("refined"))

        yield out(status_html(0, "Starting…"))
        try:
            for ev in pipe.stream(audio_path, hint or ""):
                stage = ev.stage
                d = ev.data
                raw, refined = d.get("raw", raw), d.get("refined", refined)
                if "record" in d:
                    record, diff, files = render_record_html(d["record"]), d["diff"], d["files"]
                yield out(status_html(ev.stage, ev.message))
        except PipelineError as exc:
            yield out(status_html(stage, str(exc), error=True))
        except Exception:  # noqa: BLE001
            log.exception("Unexpected pipeline failure")
            yield out(status_html(stage, "Something went wrong while processing this recording. Please try again or use a different file.", error=True))

    with gr.Blocks(title="Meeting Intelligence System", css=CSS) as demo:
        gr.Markdown("# 🧠 Meeting Intelligence System\nUpload or record an English meeting. The app transcribes it, "
                    "refines domain terms, then writes minutes, decisions and action items. "
                    "Owners and deadlines that were not stated are shown as **Unspecified**.")
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Meeting recording")
                hint = gr.Textbox(label="Domain terms (optional)", placeholder="e.g. Kubernetes, PostgreSQL, ArUco, Qwen",
                                  info="Comma-separated names/acronyms that may occur; helps spelling in transcription and refinement.")
                btn = gr.Button("Process meeting", variant="primary")
            with gr.Column(scale=2):
                status = gr.HTML(status_html(-1, "Waiting for a recording…"))
        with gr.Tabs():
            with gr.Tab("Meeting record"):
                record = gr.HTML(empty_note("The meeting record will appear here."))
            with gr.Tab("Raw transcript"):
                raw = gr.Textbox(lines=14, show_copy_button=True, label="Speech-to-text output", interactive=False)
            with gr.Tab("Refined transcript"):
                refined = gr.Textbox(lines=14, show_copy_button=True, label="After domain-aware refinement", interactive=False)
            with gr.Tab("Raw vs refined"):
                diff = gr.HTML(empty_note("The raw-vs-refined comparison will appear here."))
        with gr.Row():
            f_json = gr.File(label="meeting_record.json", interactive=False)
            f_txt = gr.File(label="meeting_record.txt", interactive=False)
            f_html = gr.File(label="meeting_report.html", interactive=False)
            f_raw = gr.File(label="raw_transcript.txt", interactive=False)
            f_ref = gr.File(label="refined_transcript.txt", interactive=False)
        btn.click(run, [audio, hint], [status, raw, refined, diff, record, f_json, f_txt, f_html, f_raw, f_ref])
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipe = MeetingPipeline()
    pipe.warmup()
    build_app(pipe).queue(max_size=4).launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")),
                                             share=cfg._flag("SHARE", "0"))
