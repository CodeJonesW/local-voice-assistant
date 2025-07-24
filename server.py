import os
import json
import subprocess
import tempfile
from io import BytesIO

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

from retriever import Retriever
from rag_prompt_builder import build_prompt

app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*")

retriever = Retriever()

_buffers = {}

def transcribe(audio_path: str) -> str:
    """Run whisper.cpp on the given audio file and return the text."""
    subprocess.run([
        "./whisper.cpp/build/bin/whisper-cli",
        "-m", "models/ggml-base.en.bin",
        "-f", audio_path,
        "-otxt",
    ], capture_output=True, text=True)
    txt_file = audio_path + ".txt"
    if os.path.exists(txt_file):
        with open(txt_file) as f:
            return f.read().strip()
    return ""

def generate_response(prompt: str) -> str:
    prompt_json = json.dumps({"prompt": prompt, "n_predict": 100})
    result = subprocess.run([
        "curl", "-s", "-X", "POST", "http://localhost:8000/completion", "-d", prompt_json
    ], capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        return data.get("content", "")
    except json.JSONDecodeError:
        return ""

def prepare_prompt(text: str) -> str:
    chunks = retriever.retrieve(text)
    if chunks:
        return build_prompt(text, chunks)
    return text

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400
    tmp = tempfile.NamedTemporaryFile(delete=False)
    file.save(tmp.name)
    retriever.add_files([tmp.name])
    os.unlink(tmp.name)
    return jsonify({"status": "ok"})

@socketio.on('connect')
def ws_connect():
    _buffers[request.sid] = BytesIO()

@socketio.on('disconnect')
def ws_disconnect():
    _buffers.pop(request.sid, None)

@socketio.on('audio_chunk')
def handle_audio(data):
    buf = _buffers.get(request.sid)
    if buf is not None:
        buf.write(data)

@socketio.on('stop')
def handle_stop():
    buf = _buffers.get(request.sid)
    if not buf:
        return
    audio_path = f"mic_{request.sid}.wav"
    with open(audio_path, 'wb') as f:
        f.write(buf.getvalue())
    text = transcribe(audio_path)
    prompt = prepare_prompt(text)
    reply = generate_response(prompt)
    emit('transcript', {'text': text})
    emit('response', {'text': reply})
    for ext in ["", ".txt"]:
        try:
            os.remove(audio_path + ext)
        except OSError:
            pass
    _buffers[request.sid] = BytesIO()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
