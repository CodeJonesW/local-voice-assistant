"""Tests for app.audio module."""

import numpy as np
from unittest.mock import patch, MagicMock

from app.audio import record, play, save


@patch("app.audio.sd")
def test_record_returns_array(mock_sd):
    fake_audio = np.zeros((16000, 1), dtype=np.float32)
    mock_sd.rec.return_value = fake_audio
    result = record(1.0, 16000)
    mock_sd.rec.assert_called_once()
    mock_sd.wait.assert_called_once()
    assert result.shape == (16000,)


@patch("app.audio.sd")
def test_play_calls_sounddevice(mock_sd):
    audio = np.zeros(16000, dtype=np.float32)
    play(audio, 16000)
    mock_sd.play.assert_called_once()
    mock_sd.wait.assert_called_once()


def test_save_creates_file(tmp_path):
    audio = np.zeros(16000, dtype=np.float32)
    path = str(tmp_path / "test.wav")
    save(audio, path, 16000)
    import os
    assert os.path.exists(path)
