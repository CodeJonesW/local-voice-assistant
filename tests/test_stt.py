"""Tests for app.stt module."""

from unittest.mock import patch, MagicMock
import numpy as np

from app.config import STTConfig


@patch("app.stt.WhisperModel")
def test_transcribe(mock_whisper_cls):
    # Setup mock
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " Hello world "
    mock_model.transcribe.return_value = ([mock_segment], None)
    mock_whisper_cls.return_value = mock_model

    # Reset cached model
    import app.stt as stt_module
    stt_module._model = None

    config = STTConfig(device="cpu")
    audio = np.zeros(16000, dtype=np.float32)
    result = stt_module.transcribe(audio, config)

    assert result == "Hello world"
    mock_model.transcribe.assert_called_once()

    # Clean up
    stt_module._model = None
