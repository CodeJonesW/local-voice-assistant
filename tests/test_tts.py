"""Tests for app.tts module."""

from unittest.mock import patch, MagicMock
import io
import wave
import struct

from app.config import TTSConfig


def _make_fake_voice():
    """Create a mock PiperVoice that writes valid WAV data."""
    mock_voice = MagicMock()

    def fake_synthesize(text, wav_file):
        # Write minimal valid WAV data
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        samples = [0] * 2205  # 0.1s of silence
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    mock_voice.synthesize.side_effect = fake_synthesize
    return mock_voice


@patch("app.tts.PiperVoice")
def test_synthesize(mock_piper_cls):
    mock_piper_cls.load.return_value = _make_fake_voice()

    import app.tts as tts_module
    tts_module._voice = None

    config = TTSConfig(model_path="fake.onnx")
    audio, sr = tts_module.synthesize("hello", config)

    assert sr == 22050
    assert len(audio) > 0

    tts_module._voice = None
