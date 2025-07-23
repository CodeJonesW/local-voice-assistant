import subprocess
import os
from TTS.api import TTS
import json

# Initialize TTS
tts_engine = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

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
        record_audio()
        text = transcribe()
        print(f"🗣️ You said: {text}")
        reply = generate_response(text)
        print(f"🤖 AI: {reply}")
        speak(reply)

if __name__ == "__main__":
    loop()
