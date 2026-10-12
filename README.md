# Meeting Intelligence System (Inter IIT Tech Meet 15.0 – ML Bootcamp)

Upload an English meeting recording → get the raw transcript, a refined transcript, structured minutes, key
decisions and action items. Owners and deadlines that were not stated are reported as **Unspecified**.

## Pipeline and model roles

```
audio ─ffmpeg (16 kHz mono, validated)─► Whisper large-v3 ──► RAW transcript
RAW ─► LLM #1  Qwen2.5-7B-Instruct (4-bit)  – domain-aware refinement, guarded ──► REFINED transcript
REFINED ─► LLM #2  Qwen2.5-14B-Instruct (4-bit) – staged extraction + grounding ──► JSON / TXT / HTML record
```

| Stage | Model | What it does | How output moves on |
|---|---|---|---|
| Speech-to-text | `openai/whisper-large-v3` | 30 s sliding windows, optional glossary prompt | raw text |
| Refinement (LLM #1) | `Qwen/Qwen2.5-7B-Instruct`, NF4 | fixes recognition errors in technical terms/names, chunk by chunk with previous-chunk context | refined text |
| Documentation (LLM #2) | `Qwen/Qwen2.5-14B-Instruct`, NF4 | minutes, decisions, action items, summary over overlapping windows | validated record |

The two LLM roles are separate stages with separate prompts (`meeting_ai/refine.py`, `meeting_ai/document.py`).
Models are configurable via env vars (`ASR_MODEL`, `REFINER_MODEL`, `DOC_MODEL`); with a smaller GPU use e.g.
`DOC_MODEL=Qwen/Qwen2.5-7B-Instruct`.

## How hallucination is controlled (in code, not only in prompts)

* **Refinement guards** – a refined chunk is accepted only if length, numbers, negation count and similarity to the raw chunk
  stay within limits; otherwise the raw chunk is kept and the reason is logged.
* **Evidence** – every decision/action item must quote the transcript; quotes that cannot be found are rejected.
* **Owners/deadlines** – must literally occur in the transcript, otherwise `Unspecified`.
* **Verifier pass** – the model re-checks each candidate: proposal ≠ decision, "someone should" ≠ task.
* **Audit** – rejected candidates and repairs are listed in `meeting_record.json` (`audit`) and the HTML report.

Output schema (identical in JSON, TXT and HTML): `summary`, `minutes[{topic, discussion}]`,
`decisions[{decision, evidence}]`, `action_items[{task, owner, deadline, evidence}]`. All normalisation lives in
`meeting_ai/schema.py`.

## Run

**Kaggle** (GPU T4 x2, Internet on):
```python
!git clone <your-repo-url> app && cd app && pip install -q -r requirements.txt
!cd app && SHARE=1 python app.py        # prints a public gradio.live link (valid while the session runs)
```
**Local / server with an NVIDIA GPU**: `pip install torch && pip install -r requirements.txt && python app.py` → http://localhost:7860
**Docker**: `docker build -t meeting-ai . && docker run --gpus all -p 7860:7860 meeting-ai`

Outputs per run: `raw_transcript.txt`, `refined_transcript.txt`, `meeting_record.json`, `meeting_record.txt`, `meeting_report.html`.
Errors (no file, empty, unsupported, corrupt, silent, no speech) are shown in the status panel.

## Tests and evaluation

```bash
pytest -q tests                                              # no GPU needed (uses a scripted fake model)
python evaluate.py asr --audio a.wav --reference a_ref.txt   # word error rate on a recording you transcribed by hand
python evaluate.py doc                                       # decision/task precision & recall + invented owner/deadline count
```
Report the numbers you actually measure; `eval/cases.json` holds proposal-vs-decision, negation, unassigned-task and
disagreement cases, and you should add cases from your own recordings.
