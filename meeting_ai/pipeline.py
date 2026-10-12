"""End-to-end orchestration: audio -> Whisper -> LLM #1 (refine) -> LLM #2 (document) -> exports.

`MeetingPipeline.stream()` is a generator that yields progress events, so a UI can show live status."""
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import config as cfg
from .audio import prepare_audio
from .document import generate_record_iter
from .errors import PipelineError
from .exporters import diff_html, write_outputs
from .refine import refine_transcript_iter
from .textutils import wc

STAGES = ("Prepare audio", "Transcribe", "Refine", "Document")


@dataclass
class Event:
    stage: int                      # 0..3 = running that stage, 4 = finished
    message: str
    data: dict = field(default_factory=dict)


def fmt_duration(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class MeetingPipeline:
    def __init__(self):
        import torch
        from .models import LLM, Transcriber
        if not torch.cuda.is_available():
            raise PipelineError("No GPU detected. This app needs a CUDA GPU (Kaggle: GPU T4 x2).")
        n = torch.cuda.device_count()
        self.resident = n >= 2        # two GPUs: keep every model loaded; one GPU: load/unload per stage
        self.asr = Transcriber(cfg.ASR_MODEL, 0, cfg.ASR_BATCH_SIZE)
        self.refiner = LLM(cfg.REFINER_MODEL, 0, cfg.LOAD_4BIT)
        self.doc = LLM(cfg.DOC_MODEL, 1 if n > 1 else 0, cfg.LOAD_4BIT)

    def warmup(self):
        if self.resident:
            for m in (self.asr, self.refiner, self.doc):
                m.load()

    @contextmanager
    def _use(self, model):
        model.load()
        try:
            yield model
        finally:
            if not self.resident:
                model.unload()

    def stream(self, audio_path, hint_terms=""):
        t0 = time.time()
        yield Event(0, "Checking and converting the audio…")
        wav, duration = prepare_audio(audio_path, cfg.WORK_DIR)          # raises AudioError
        yield Event(0, f"Audio ready ({fmt_duration(duration)}).", {"duration": duration})

        yield Event(1, "Transcribing speech (first run loads the model)…")
        try:
            with self._use(self.asr) as asr:
                raw = asr.transcribe(wav, hint_terms)
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"Transcription failed: {exc}") from exc
        if wc(raw) < 3:
            raise PipelineError("No intelligible speech was detected in the recording.")
        yield Event(1, "Transcription complete.", {"raw": raw})

        yield Event(2, "Refining the transcript…")
        refine_report, refined = [], raw
        try:
            with self._use(self.refiner) as llm:
                for p in refine_transcript_iter(llm, raw, hint_terms):
                    refined, refine_report = p["text"], p["report"]
                    yield Event(2, f"Refining the transcript (part {p['i']} of {p['n']})…", {"refined": refined})
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"Transcript refinement failed: {exc}") from exc

        yield Event(3, "Generating minutes, decisions and action items…", {"refined": refined})
        try:
            with self._use(self.doc) as llm:
                for p in generate_record_iter(llm, refined):
                    if p.get("done"):
                        record, audit = p["record"], p["audit"]
                    else:
                        yield Event(3, p["message"])
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"Documentation step failed: {exc}") from exc

        audit["refinement_fallbacks"] = [r for r in refine_report if r["fallback"]]
        meta = {"duration": duration, "duration_text": fmt_duration(duration), "words": wc(raw),
                "asr_model": cfg.ASR_MODEL, "refiner_model": cfg.REFINER_MODEL, "doc_model": cfg.DOC_MODEL,
                "seconds_total": round(time.time() - t0, 1)}
        audit["meta"] = meta
        files = write_outputs(f"{cfg.WORK_DIR}/run_{uuid.uuid4().hex[:8]}", record, raw, refined, audit, meta)
        yield Event(4, "Done.", {"raw": raw, "refined": refined, "record": record, "audit": audit,
                                 "diff": diff_html(raw, refined), "files": files, "meta": meta})
