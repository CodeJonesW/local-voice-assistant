"""Download required models for the voice assistant."""

import os
import sys
from pathlib import Path

import requests


MODELS = {
    "llm": {
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "path": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "Mistral 7B Instruct v0.2 (Q4_K_M quantization)",
    },
    "tts_model": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "path": "models/en_US-lessac-medium.onnx",
        "description": "Piper TTS voice: en_US-lessac-medium",
    },
    "tts_config": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "path": "models/en_US-lessac-medium.onnx.json",
        "description": "Piper TTS voice config",
    },
}


def download_file(url: str, dest: str) -> None:
    """Download a file with progress display."""
    print(f"  Downloading: {url}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)
    print()


def main():
    """Download all required models."""
    print("Local Voice Assistant - Model Downloader")
    print("=" * 40)

    # faster-whisper downloads its own models automatically,
    # so we only need LLM and TTS models.

    for name, info in MODELS.items():
        dest = info["path"]
        if os.path.exists(dest):
            print(f"[skip] {info['description']} (already exists)")
            continue
        print(f"[download] {info['description']}")
        try:
            download_file(info["url"], dest)
            print(f"  Saved to: {dest}")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  You can manually download from: {info['url']}")

    print()
    print("Done! faster-whisper will download its model on first run.")
    print()
    print("To start the assistant:")
    print("  python -m app.assistant            # CLI mode")
    print("  python -m app.assistant --text-only # Text-only mode")
    print("  python -m app.server               # Web UI mode")


if __name__ == "__main__":
    main()
