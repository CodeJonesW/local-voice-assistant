"""Tests for app.llm module."""

from unittest.mock import patch, MagicMock

from app.config import LLMConfig


@patch("app.llm.Llama")
def test_generate(mock_llama_cls):
    mock_model = MagicMock()
    mock_model.return_value = {
        "choices": [{"text": " This is a response. "}]
    }
    mock_llama_cls.return_value = mock_model

    import app.llm as llm_module
    llm_module._model = None

    config = LLMConfig(model_path="fake.gguf")
    result = llm_module.generate("Hello", config)

    assert result == "This is a response."

    llm_module._model = None


@patch("app.llm.detect_hardware", return_value="cpu")
@patch("app.llm.Llama")
def test_cpu_forces_zero_gpu_layers(mock_llama_cls, mock_hw):
    mock_model = MagicMock()
    mock_llama_cls.return_value = mock_model

    import app.llm as llm_module
    llm_module._model = None

    config = LLMConfig(model_path="fake.gguf", n_gpu_layers=-1)
    llm_module.load_model(config)

    # Should have been called with n_gpu_layers=0
    call_kwargs = mock_llama_cls.call_args[1]
    assert call_kwargs["n_gpu_layers"] == 0

    llm_module._model = None
