"""Audio validation and conversion to 16 kHz mono WAV. Every failure raises AudioError with a clear message."""
import os
import shutil
import subprocess
import uuid

from .errors import AudioError

SUPPORTED_EXTS = (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".mp4", ".webm", ".mkv", ".opus", ".amr")
MIN_SECONDS = 1.0
SILENCE_RMS = 1e-3


def prepare_audio(src_path, work_dir):
    """Validate the upload and convert it. Returns (wav_path, duration_seconds)."""
    if not src_path:
        raise AudioError("No audio file was provided. Please upload or record a meeting first.")
    if not os.path.isfile(src_path):
        raise AudioError("The audio file could not be found on the server. Please upload it again.")
    if os.path.getsize(src_path) == 0:
        raise AudioError("The uploaded file is empty (0 bytes).")
    ext = os.path.splitext(src_path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise AudioError(f"Unsupported file type '{ext or 'unknown'}'. Supported: {', '.join(SUPPORTED_EXTS)}.")
    if shutil.which("ffmpeg") is None:
        raise AudioError("ffmpeg is not installed on the server, so audio cannot be decoded.")

    os.makedirs(work_dir, exist_ok=True)
    dst = os.path.join(work_dir, f"{uuid.uuid4().hex}.wav")
    proc = subprocess.run(["ffmpeg", "-y", "-i", src_path, "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", dst],
                          capture_output=True)
    if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) <= 44:
        raise AudioError("This file could not be decoded as audio. It may be corrupt or not an audio recording.")

    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(dst, dtype="float32")
    duration = len(audio) / float(sr)
    if duration < MIN_SECONDS:
        raise AudioError(f"The recording is too short ({duration:.1f} s) to contain a meeting.")
    if float(np.sqrt(np.mean(np.square(audio)))) < SILENCE_RMS:
        raise AudioError("The recording appears to be silent. Please check the file and try again.")
    return dst, duration
