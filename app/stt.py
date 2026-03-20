"""Speech-to-text using faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel

from app.config import STTConfig, detect_hardware

_model: WhisperModel | None = None


def _get_device(config: STTConfig) -> str:
    """Resolve 'auto' device setting."""
    if config.device != "auto":
        return config.device
    hw = detect_hardware()
    if hw == "cuda":
        return "cuda"
    return "cpu"


def load_model(config: STTConfig) -> WhisperModel:
    """Load the whisper model (cached after first call)."""
    global _model
    if _model is None:
        device = _get_device(config)
        compute_type = config.compute_type
        # float16 not supported on CPU
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
        _model = WhisperModel(
            config.model_size,
            device=device,
            compute_type=compute_type,
        )
    return _model


def transcribe(audio: np.ndarray, config: STTConfig) -> str:
    """Transcribe audio array to text.

    Args:
        audio: 1D float32 numpy array at 16kHz.
        config: STT configuration.

    Returns:
        Transcribed text string.
    """
    model = load_model(config)
    segments, _ = model.transcribe(audio, beam_size=5)
    return " ".join(segment.text.strip() for segment in segments).strip()
