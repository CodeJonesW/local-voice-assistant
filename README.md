# 🧠 Local Voice Assistant (Offline, GPU-Accelerated)

This is a fully offline voice assistant pipeline running on Linux with GPU acceleration. It uses:
- 🎙️ `whisper.cpp` for speech-to-text
- 🦙 `llama.cpp` with TinyLlama for language generation
- 🔊 Coqui TTS for speech synthesis

Everything runs **locally**, with no API calls or internet access required after setup.

---

## 🗂️ Project Structure

```bash
.
├── assistant.py                # Main Python loop for record>transcribe>generate>speak
├── llama.cpp/                  # Local LLM server (GGUF support)
├── whisper.cpp/                # Local Whisper engine for STT
├── models/
│   ├── ggml-base.en.bin        # Whisper base English model
│   └── tinyllama-1.1b-chat...  # TinyLlama quantized GGUF model
├── mic.wav                     # Temporary recorded audio
├── mic.wav.txt                 # Transcribed text
├── output.wav                  # Spoken response
├── run.sh                      # Optional launcher
├── retriever.py                # Simple vector store for RAG
├── rag_prompt_builder.py       # Helper to build RAG prompts
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### ✅ Setup Instructions
Clone and install each component in order:

1. 🧠 Clone This Project
```bash
git clone https://github.com/YOUR_USERNAME/local-voice-assistant.git
cd local-voice-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 🎙️ Install whisper.cpp (speech-to-text)
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
cd ..
```

Download the Whisper model:
```bash
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -P models/
```

3. 🦙 Install llama.cpp and Run the LLM Server

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=on
make -j
cd ../..
```

Download the TinyLlama GGUF model (Q4_K_M quantization):
```bash
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -P models/
```

Run the LLM server:

./llama.cpp/build/bin/llama-server -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --port 8000


4. 🔊 Install Coqui TTS (Text-to-Speech)
```bash
pip install TTS
```

The Python script will automatically load the tts_models/en/ljspeech/tacotron2-DDC model on first run.

### 🚀 Running the Assistant
Once the llama-server is running on port 8000, in a separate terminal:

```bash
source .venv/bin/activate
python assistant.py
```

This will:
- Record 5 seconds of audio
- Transcribe it using Whisper
- Generate a response using TinyLlama
- Speak the response with Coqui TTS

### 📚 Custom Knowledge (RAG mode)
Add your own documents to the vector store and run the assistant in custom mode:

```bash
python retriever.py --add my_notes.txt
python assistant.py --mode custom
```

Or add a file on the fly when launching the assistant:

```bash
python assistant.py --add-doc my_notes.txt --mode custom
```

To bulk train on all files in the `toTrain` directory and then move them to
`previouslyTrainedOn`:

```bash
python retriever.py --train-folder toTrain
```



### 🙌 Acknowledgements
- ggerganov/whisper.cpp - https://github.com/ggml-org/whisper.cpp
- ggerganov/llama.cpp - https://github.com/ggml-org/llama.cpp
- coqui-ai/TTS - https://github.com/coqui-ai/TTS
- TheBloke on Hugging Face - https://huggingface.co/TheBloke