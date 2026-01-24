#!/bin/bash

# --- Configuration ---
PORT="6767"
DEFAULT_CTX="8192"

export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_DEBUG=1
export OLLAMA_FLASH_ATTENTION=1

# 1. Check for Ollama CLI
if ! command -v ollama &> /dev/null; then
    echo "Error: 'ollama' command not found."
    echo "Please install Ollama from https://ollama.com"
    exit 1
fi

# 2. Start/Ensure Server is Running
echo "--- Ollama Server Setup ---"

if ! curl -s "http://127.0.0.1:$PORT" > /dev/null; then
    echo "Starting Ollama server on port $PORT..."
    ollama serve &
    SERVER_PID=$!
    
    echo "Waiting for server to become responsive..."
    while ! curl -s "http://127.0.0.1:$PORT" > /dev/null; do
        sleep 1
    done
else
    echo "Ollama server is already running on port $PORT."
fi

# 3. Model Selection
echo ""
echo "Select your base model (e.g., llama3.1, mistral, deepseek-r1):"
read -p "Model Name (default: gpt-oss:20b): " BASE_MODEL
BASE_MODEL=${BASE_MODEL:-gpt-oss:20b}

# 4. Context Size Selection
read -p "Enter context length (default: $DEFAULT_CTX): " ctx_input
CTX_LENGTH=${ctx_input:-$DEFAULT_CTX}

# 5. Pull & Create Custom Model
echo "Checking/Pulling base model '$BASE_MODEL'..."
ollama pull "$BASE_MODEL"

echo "Creating 'TestAgent-model' with context window size $CTX_LENGTH..."
# Create a temporary Modelfile to set the parameter
echo "FROM $BASE_MODEL" > Modelfile.temp
echo "PARAMETER num_ctx $CTX_LENGTH" >> Modelfile.temp
echo "PARAMETER temperature 0.6" >> Modelfile.temp
echo "PARAMETER top_k 20" >> Modelfile.temp

# Create the model alias with the config
ollama create TestAgent-model -f Modelfile.temp
rm Modelfile.temp

echo ""
echo "Server is ready at http://127.0.0.1:$PORT"
echo "Model 'TestAgent-model' is active (Context: $CTX_LENGTH)."
echo "(Press Ctrl+C to stop watching this script)"

# 6. Keep script alive
if [ -n "$SERVER_PID" ]; then
    wait $SERVER_PID
else
    tail -f /dev/null
fi