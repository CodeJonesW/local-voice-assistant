#!/bin/bash

set -e

LLAMA_PORT=8000
LLAMA_MODEL="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LLAMA_SERVER="./llama.cpp/build/bin/llama-server"
VENV_ACTIVATE=".venv/bin/activate"

# Step 1: Kill process using port 8000 if it exists
echo "🔎 Checking if port $LLAMA_PORT is in use..."
PID_ON_PORT=$(lsof -ti tcp:$LLAMA_PORT || true)
if [ -n "$PID_ON_PORT" ]; then
  echo "⚠️ Port $LLAMA_PORT is in use. Killing PID(s): $PID_ON_PORT"
  kill -9 $PID_ON_PORT
  sleep 1
fi

# Step 2: Start llama-server in background
echo "🦙 Starting llama-server..."
$LLAMA_SERVER -m "$LLAMA_MODEL" --port $LLAMA_PORT > llama.log 2>&1 &
LLAMA_PID=$!
echo "✅ llama-server started with PID $LLAMA_PID"

# Step 3: Wait for llama-server /completion route to respond
echo -n "⏳ Waiting for llama-server to respond at /completion"
until curl -s -X POST "http://localhost:$LLAMA_PORT/completion" -d '{"prompt": "Hello", "n_predict": 1}' | grep -q "content"; do
  echo -n "."
  sleep 1
done
echo " ✅"

# Step 4: Activate Python virtual environment
echo "📦 Activating virtual environment..."
source "$VENV_ACTIVATE"

# Step 5: Run the assistant
echo "🎤 Running assistant.py..."
python assistant.py

# Step 6: Cleanup
echo "🛑 Stopping llama-server..."
kill $LLAMA_PID
