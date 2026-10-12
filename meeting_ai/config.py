"""Central configuration. Every value can be overridden with an environment variable."""
import os
import tempfile


def _flag(name, default="1"):
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "")


# ---- models (three distinct roles) -------------------------------------------------
ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-large-v3")            # speech-to-text
REFINER_MODEL = os.getenv("REFINER_MODEL", "Qwen/Qwen2.5-7B-Instruct")   # LLM #1: transcript refinement
DOC_MODEL = os.getenv("DOC_MODEL", "Qwen/Qwen2.5-14B-Instruct")          # LLM #2: documentation
LOAD_4BIT = _flag("LOAD_4BIT")
ASR_BATCH_SIZE = int(os.getenv("ASR_BATCH_SIZE", "4"))

# ---- refinement guards -------------------------------------------------------------
REFINE_CHUNK_WORDS = int(os.getenv("REFINE_CHUNK_WORDS", "220"))
REFINE_CONTEXT_WORDS = 40
REFINE_MIN_RATIO = 0.85        # refined/raw word-count ratio must stay inside this band
REFINE_MAX_RATIO = 1.15
REFINE_MIN_SIMILARITY = 0.75   # word-level similarity between raw and refined chunk

# ---- documentation stage -----------------------------------------------------------
DOC_WINDOW_WORDS = int(os.getenv("DOC_WINDOW_WORDS", "900"))
DOC_OVERLAP_WORDS = int(os.getenv("DOC_OVERLAP_WORDS", "120"))
EVIDENCE_THRESHOLD = 0.8       # min similarity for a quoted piece of evidence to count as "found"
VERIFY_WITH_LLM = _flag("VERIFY_WITH_LLM")
JSON_RETRIES = 2

WORK_DIR = os.getenv("WORK_DIR", os.path.join(tempfile.gettempdir(), "meeting_ai"))
