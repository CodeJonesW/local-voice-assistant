import subprocess
import os
import json
import argparse
from TTS.api import TTS

from metrics import MetricsLogger

# Initialize TTS
tts_engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

# Parse CLI args / env var for metrics logging
parser = argparse.ArgumentParser()
parser.add_argument("--log-metrics", action="store_true", default=os.getenv("LOG_METRICS") == "1",
                    help="Enable metrics logging")
args = parser.parse_args()

# Create metrics logger
metrics = MetricsLogger(enabled=args.log_metrics)

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
        with metrics.time("generate_response"):
            reply = generate_response(text)
        print(f"🤖 AI: {reply}")
        with metrics.time("speak"):
            speak(reply)
        metrics.log_run()

if __name__ == "__main__":
    loop()
