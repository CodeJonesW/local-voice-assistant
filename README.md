# Local Voice Assistant (Offline, Cross-Platform)

A fully offline voice assistant that runs on **macOS and Linux**. No API calls or internet required after setup.

**Stack:**
- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-based Whisper)
- **LLM**: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) with Mistral 7B
- **TTS**: [piper-tts](https://github.com/rhasspy/piper) (fast ONNX-based synthesis)
- **RAG**: [ChromaDB](https://github.com/chroma-core/chroma) with built-in embeddings
- **Audio**: [sounddevice](https://python-sounddevice.readthedocs.io/) (cross-platform PortAudio)

**Hardware auto-detection**: Apple Silicon (Metal), NVIDIA (CUDA), or CPU-only — no configuration needed.

---

## Project Structure

```
├── pyproject.toml              # Dependencies and project config
├── config.yaml                 # User-editable settings
├── app/
│   ├── config.py               # Settings + hardware detection
│   ├── audio.py                # Cross-platform record/play
│   ├── stt.py                  # Speech-to-text (faster-whisper)
│   ├── tts.py                  # Text-to-speech (piper)
│   ├── llm.py                  # LLM inference (llama-cpp-python)
│   ├── retriever.py            # Vector store (ChromaDB)
│   ├── rag.py                  # RAG prompt builder
│   ├── metrics.py              # Timing metrics
│   ├── assistant.py            # CLI voice assistant
│   └── server.py               # Web UI server
├── static/                     # Web UI (HTML/JS)
├── scripts/
│   └── download_models.py      # Model downloader
└── tests/                      # Test suite
```

## Setup

```bash
git clone https://github.com/CodeJonesW/local-voice-assistant.git
cd local-voice-assistant
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -e .
```

### Download Models

```bash
python -m scripts.download_models
```

This downloads:
- **Mistral 7B Instruct** (Q4_K_M, ~4.4 GB) — LLM
- **Piper en_US-lessac-medium** (~60 MB) — TTS voice
- faster-whisper downloads its model automatically on first run

## Usage

### CLI Voice Assistant
```bash
python -m app.assistant              # Voice mode (mic + speakers)
python -m app.assistant --text-only  # Text-only mode
```

### Web UI
```bash
python -m app.server
```
Open http://localhost:5000 in your browser.

### RAG Mode (Custom Knowledge)
```bash
# Add a document
python -m app.assistant --add-doc my_notes.txt

# Process all files in toTrain/ folder
python -m app.assistant --train-folder

# Run with RAG
python -m app.assistant --mode custom
```

## Configuration

Edit `config.yaml` to customize audio devices, model paths, recording duration, etc. All settings have sensible defaults.

## Acknowledgements
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [rhasspy/piper](https://github.com/rhasspy/piper)
- [chroma-core/chroma](https://github.com/chroma-core/chroma)
- [TheBloke](https://huggingface.co/TheBloke) on Hugging Face
