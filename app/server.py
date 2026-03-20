"""Flask web server for the voice assistant."""

import io
import os
import tempfile

import soundfile as sf
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

from app.config import load_config
from app.retriever import Retriever
from app.rag import build_prompt
from app import stt, llm

_config = load_config()
_retriever = Retriever(_config.retriever)

app = Flask(__name__, static_folder="../static")
socketio = SocketIO(app, cors_allowed_origins="*")

_buffers: dict[str, io.BytesIO] = {}


def _audio_bytes_to_wav(raw_bytes: bytes) -> str:
    """Save raw audio bytes to a temp WAV file and return the path.

    Browser MediaRecorder sends webm/opus by default. We write the raw
    bytes to a temp file and attempt to read with soundfile (which handles
    many formats via libsndfile). If that fails, we write as-is and let
    faster-whisper handle it.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        # Try to decode and re-encode as proper WAV
        data, sr = sf.read(io.BytesIO(raw_bytes))
        sf.write(tmp.name, data, sr, format="WAV")
    except Exception:
        # Fallback: write raw bytes and hope for the best
        tmp.write(raw_bytes)
    tmp.close()
    return tmp.name


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    file.save(tmp.name)
    tmp.close()
    _retriever.add_files([tmp.name])
    os.unlink(tmp.name)
    return jsonify({"status": "ok"})


@socketio.on("connect")
def ws_connect():
    _buffers[request.sid] = io.BytesIO()


@socketio.on("disconnect")
def ws_disconnect():
    _buffers.pop(request.sid, None)


@socketio.on("audio_chunk")
def handle_audio(data):
    buf = _buffers.get(request.sid)
    if buf is not None:
        buf.write(data)


@socketio.on("stop")
def handle_stop():
    buf = _buffers.get(request.sid)
    if not buf:
        return

    # Convert browser audio to WAV and transcribe
    wav_path = _audio_bytes_to_wav(buf.getvalue())
    try:
        audio_data, sr = sf.read(wav_path, dtype="float32")
        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        text = stt.transcribe(audio_data, _config.stt)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    # Generate response with optional RAG
    chunks = _retriever.retrieve(text)
    prompt = build_prompt(text, chunks) if chunks else text
    reply = llm.generate(prompt, _config.llm)

    emit("transcript", {"text": text})
    emit("response", {"text": reply})
    _buffers[request.sid] = io.BytesIO()


def main():
    """Entry point for the web server."""
    config = load_config()

    # Pre-load models
    print("Loading models...")
    stt.load_model(config.stt)
    llm.load_model(config.llm)
    print("Models loaded. Server starting...")

    socketio.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
