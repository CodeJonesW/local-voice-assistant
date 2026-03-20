"""Tests for app.config module."""

import platform
from unittest.mock import patch

from app.config import AppConfig, load_config, detect_hardware


def test_default_config():
    config = AppConfig()
    assert config.audio.sample_rate == 16000
    assert config.llm.max_tokens == 256
    assert config.server.port == 5000


def test_load_config_from_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "audio:\n  sample_rate: 44100\nllm:\n  max_tokens: 512\n"
    )
    config = load_config(str(cfg_file))
    assert config.audio.sample_rate == 44100
    assert config.llm.max_tokens == 512
    # Defaults preserved for unspecified values
    assert config.server.port == 5000


def test_load_config_missing_file():
    config = load_config("nonexistent.yaml")
    assert config.audio.sample_rate == 16000


@patch("app.config.platform")
def test_detect_hardware_mac(mock_platform):
    mock_platform.system.return_value = "Darwin"
    mock_platform.machine.return_value = "arm64"
    assert detect_hardware() == "metal"


@patch("app.config.platform")
def test_detect_hardware_cpu_fallback(mock_platform):
    mock_platform.system.return_value = "Windows"
    mock_platform.machine.return_value = "AMD64"
    assert detect_hardware() == "cpu"
