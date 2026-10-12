import pytest

from meeting_ai.audio import prepare_audio
from meeting_ai.errors import AudioError


def test_no_file(tmp_path):
    with pytest.raises(AudioError, match="No audio"):
        prepare_audio(None, str(tmp_path))


def test_empty_file(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"")
    with pytest.raises(AudioError, match="empty"):
        prepare_audio(str(p), str(tmp_path))


def test_unsupported_extension(tmp_path):
    p = tmp_path / "notes.pdf"
    p.write_bytes(b"%PDF")
    with pytest.raises(AudioError, match="Unsupported"):
        prepare_audio(str(p), str(tmp_path))
