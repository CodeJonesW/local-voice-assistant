import subprocess
import os
import json
import argparse
from TTS.api import TTS

from metrics import MetricsLogger
from retriever import Retriever
from rag_prompt_builder import build_prompt

# Initialize TTS
tts_engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Parse CLI args / env var for metrics logging and mode
parser = argparse.ArgumentParser()
parser.add_argument("--log-metrics", action="store_true", default=os.getenv("LOG_METRICS") == "1",
                    help="Enable metrics logging")
parser.add_argument("--mode", choices=["base", "custom"], default="base", help="Choose assistant mode")
parser.add_argument("--add-doc", help="Add a text file to the vector store and exit")
parser.add_argument("--db", default="vector_store.pkl", help="Path to vector store DB")
args = parser.parse_args()

# Create metrics logger
metrics = MetricsLogger(enabled=args.log_metrics)

# Optional retriever
retriever = Retriever(db_path=args.db)
if args.add_doc:
    retriever.add_files([args.add_doc])
    print(f"Added {args.add_doc} to {args.db}")
    raise SystemExit


def record_audio():
    print("🎤 Speak now...")
    subprocess.run(["arecord", "-f", "cd", "-d", "5", "mic.wav", "-q"])


def transcribe():
    subprocess.run([
        "./whisper.cpp/build/bin/whisper-cli",
        "-m", "models/ggml-base.en.bin",
        "-f", "mic.wav",
        "-otxt"
    ], capture_output=True, text=True)
    with open("mic.wav.txt") as f:
        return f.read().strip()


def prepare_prompt(text: str) -> str:
    if args.mode == "custom":
        chunks = retriever.retrieve(text)
        if chunks:
            return build_prompt(text, chunks)
    return text


def generate_response(prompt):
    # Escape the prompt to be safe inside JSON
    prompt_json = json.dumps({"prompt": prompt, "n_predict": 100})

    # Run the curl subprocess
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", "http://localhost:8000/completion", "-d", prompt_json],
        capture_output=True, text=True
    )

    # Parse the response and extract 'content'
    try:
        response = json.loads(result.stdout)
        return response.get("content", "Sorry, no response.")
    except json.JSONDecodeError:
        return "Sorry, could not parse the model response."


def speak(text):
    if not text.strip():
        print("🤖 (No response to speak)")
        return
    print(f"🤖 AI says: {text}")
    tts_engine.tts_to_file(text=text, file_path="output.wav")
    subprocess.run(["aplay", "output.wav"])


def loop():
    while True:
        with metrics.time("record_audio"):
            record_audio()
        with metrics.time("transcribe"):
            text = transcribe()
        print(f"🗣️ You said: {text}")
        prompt = prepare_prompt(text)
        with metrics.time("generate_response"):
            reply = generate_response(prompt)
        print(f"🤖 AI: {reply}")
        with metrics.time("speak"):
            speak(reply)
        metrics.log_run()


if __name__ == "__main__":
    loop()
