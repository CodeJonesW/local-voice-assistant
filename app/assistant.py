"""CLI voice assistant — main loop."""

import argparse
import os

from app.config import load_config
from app.metrics import MetricsLogger
from app.retriever import Retriever
from app.rag import build_prompt
from app import audio, stt, tts, llm


def main():
    parser = argparse.ArgumentParser(description="Local Voice Assistant")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--log-metrics", action="store_true",
        default=os.getenv("LOG_METRICS") == "1",
        help="Enable metrics logging",
    )
    parser.add_argument(
        "--mode", choices=["base", "custom"], default="base",
        help="Assistant mode (base=direct, custom=RAG)",
    )
    parser.add_argument(
        "--add-doc", help="Add a text file to the vector store and exit",
    )
    parser.add_argument(
        "--train-folder", action="store_true",
        help="Process all files in the train folder and exit",
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="Text input/output only (no microphone or speakers)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.log_metrics:
        config.metrics.enabled = True

    # Handle document management commands
    retriever = Retriever(config.retriever)
    if args.add_doc:
        retriever.add_files([args.add_doc])
        print(f"Added {args.add_doc} to vector store")
        return
    if args.train_folder:
        retriever.add_folder()
        print("Processed training folder")
        return

    metrics = MetricsLogger(config.metrics)

    # Pre-load models
    print("Loading models...")
    if not args.text_only:
        stt.load_model(config.stt)
        tts.load_voice(config.tts)
    llm.load_model(config.llm)
    print("Models loaded. Ready!")

    while True:
        try:
            if args.text_only:
                text = input("You: ")
            else:
                with metrics.time("record_audio"):
                    audio_data = audio.record(
                        config.audio.record_duration,
                        config.audio.sample_rate,
                        config.audio.input_device,
                    )
                with metrics.time("transcribe"):
                    text = stt.transcribe(audio_data, config.stt)
                print(f"You: {text}")

            if not text.strip():
                continue
            if text.strip().lower() in {"exit", "quit"}:
                print("Exiting.")
                break

            # Prepare prompt (with RAG if in custom mode)
            with metrics.time("prepare_prompt"):
                if args.mode == "custom":
                    chunks = retriever.retrieve(text)
                    if chunks:
                        print(f"Retrieved {len(chunks)} chunks")
                        prompt = build_prompt(text, chunks)
                    else:
                        prompt = text
                else:
                    prompt = text

            with metrics.time("generate_response"):
                reply = llm.generate(prompt, config.llm)
            print(f"AI: {reply}")

            if not args.text_only and reply.strip():
                with metrics.time("speak"):
                    audio_data, sr = tts.synthesize(reply, config.tts)
                    audio.play(audio_data, sr, config.audio.output_device)

            metrics.log_run()

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
