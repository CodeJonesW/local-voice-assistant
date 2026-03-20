"""Configuration management with hardware auto-detection."""

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    record_duration: int = 5
    input_device: Optional[int] = None
    output_device: Optional[int] = None


class STTConfig(BaseModel):
    model_size: str = "base.en"
    device: str = "auto"
    compute_type: str = "int8"


class TTSConfig(BaseModel):
    model_path: str = "models/en_US-lessac-medium.onnx"


class LLMConfig(BaseModel):
    model_path: str = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    max_tokens: int = 256
    temperature: float = 0.7
    n_gpu_layers: int = -1
    context_size: int = 2048


class RetrieverConfig(BaseModel):
    db_path: str = "chroma_db"
    chunk_size: int = 200
    n_results: int = 3
    train_folder: str = "toTrain"
    processed_folder: str = "previouslyTrainedOn"


class MetricsConfig(BaseModel):
    enabled: bool = False
    path: str = "metrics.csv"
    rolling_window: int = 5


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 5000


class AppConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    stt: STTConfig = STTConfig()
    tts: TTSConfig = TTSConfig()
    llm: LLMConfig = LLMConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    metrics: MetricsConfig = MetricsConfig()
    server: ServerConfig = ServerConfig()


def detect_hardware() -> str:
    """Detect available hardware acceleration.

    Returns one of: 'metal', 'cuda', 'cpu'
    """
    system = platform.system()

    # macOS with Apple Silicon
    if system == "Darwin" and platform.machine() == "arm64":
        return "metal"

    # Check for NVIDIA GPU
    if system == "Linux":
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return "cuda"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return "cpu"


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from YAML file, falling back to defaults."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()
