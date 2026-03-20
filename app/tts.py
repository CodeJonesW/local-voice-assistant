"""Text-to-speech using piper-tts."""

import io
import wave

import numpy as np
from piper import PiperVoice

from app.config import TTSConfig

_voice: PiperVoice | None = None


def load_voice(config: TTSConfig) -> PiperVoice:
    """Load the piper voice model (cached after first call)."""
    global _voice
    if _voice is None:
        _voice = PiperVoice.load(config.model_path)
    return _voice


def synthesize(text: str, config: TTSConfig) -> tuple[np.ndarray, int]:
    """Synthesize text to audio.

    Returns:
        Tuple of (audio array, sample_rate).
    """
    voice = load_voice(config)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        voice.synthesize(text, wav)
    buf.seek(0)
    with wave.open(buf, "rb") as wav:
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


def synthesize_to_file(text: str, path: str, config: TTSConfig) -> None:
    """Synthesize text and save to a WAV file."""
    voice = load_voice(config)
    with wave.open(path, "wb") as wav:
        voice.synthesize(text, wav)
