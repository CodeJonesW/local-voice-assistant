"""Cross-platform audio recording and playback using sounddevice."""

import numpy as np
import sounddevice as sd
import soundfile as sf


def record(duration: float, sample_rate: int = 16000,
           device: int | None = None) -> np.ndarray:
    """Record audio from the microphone.

    Returns a 1D float32 numpy array of audio samples.
    """
    print("Speak now...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.flatten()


def play(audio: np.ndarray, sample_rate: int = 16000,
         device: int | None = None) -> None:
    """Play audio data through the speakers."""
    sd.play(audio, samplerate=sample_rate, device=device)
    sd.wait()


def play_file(path: str, device: int | None = None) -> None:
    """Play a WAV file through the speakers."""
    audio, sample_rate = sf.read(path, dtype="float32")
    play(audio, sample_rate, device)


def save(audio: np.ndarray, path: str, sample_rate: int = 16000) -> None:
    """Save audio data to a WAV file."""
    sf.write(path, audio, sample_rate)
