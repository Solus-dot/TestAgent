#!/bin/bash

# --- Configuration ---
PORT="6767"
DEFAULT_CTX="8192"

# 1. Check for LMS CLI
if ! command -v lms &> /dev/null; then
    echo "Error: 'lms' command not found."
    echo "Please install LM Studio and run: 'lms bootstrap'"
    exit 1
fi

# 2. Start/Ensure Server is Running
echo "--- LM Studio Server Setup ---"
lms server start --port "$PORT" > /dev/null 2>&1

# 3. Unload old models (Clean slate)
lms unload --all > /dev/null 2>&1

# 4. Ask for Context Length
read -p "Enter context length (default: $DEFAULT_CTX): " ctx_input
CTX_LENGTH=${ctx_input:-$DEFAULT_CTX}

# 5. Load Model (Interactive)
# We run 'lms load' without a model name, which triggers the menu.
# We pass the flags so they apply to whatever model you pick.
echo ""
echo "Select your model from the menu below:"
echo "-------------------------------------"

lms load \
    --context-length "$CTX_LENGTH" \
    --gpu max \
    --identifier "TestAgent-Model"

# 6. Stream Logs
if [ $? -eq 0 ]; then
    echo ""
    echo "Server is ready at http://127.0.0.1:$PORT"
    echo "(Ctrl+C will stop watching logs, but server keeps running)"
    echo "--- Server Logs ---"
    lms log stream --source model --stats --json --filter input,output | uv run src/format_logs.py
else
    echo "Failed to load model."
fi