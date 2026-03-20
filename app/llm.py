"""LLM inference using llama-cpp-python."""

from llama_cpp import Llama

from app.config import LLMConfig, detect_hardware

_model: Llama | None = None


def load_model(config: LLMConfig) -> Llama:
    """Load the LLM (cached after first call)."""
    global _model
    if _model is None:
        n_gpu_layers = config.n_gpu_layers
        hw = detect_hardware()
        # On CPU-only systems, force 0 GPU layers
        if hw == "cpu":
            n_gpu_layers = 0
        _model = Llama(
            model_path=config.model_path,
            n_ctx=config.context_size,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    return _model


def generate(prompt: str, config: LLMConfig) -> str:
    """Generate a response from the LLM.

    Uses Mistral instruct format for chat prompting.
    """
    model = load_model(config)
    formatted = f"[INST] {prompt} [/INST]"
    result = model(
        formatted,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=["</s>", "[INST]"],
    )
    return result["choices"][0]["text"].strip()
