#!/bin/bash

# Config
MODEL_DIR="$HOME/.lmstudio/models"
HOST="127.0.0.1"
PORT="6767"
DEFAULT_CTX="8192"

echo "Scanning for .gguf models in $MODEL_DIR..."
models=()
while IFS= read -r -d '' file; do
    models+=("$file")
done < <(find "$MODEL_DIR" -type f -name "*.gguf" -print0)

# Check if we found any models
if [ ${#models[@]} -eq 0 ]; then
    echo "Error: No .gguf files found in $MODEL_DIR"
    exit 1
fi

echo "Found the following models:"
for i in "${!models[@]}"; do
    # Just show the filename, not the full path, for readability
    filename=$(basename "${models[$i]}")
    echo "[$((i+1))] $filename"
done

echo ""
read -p "Select a model (enter number): " selection
index=$((selection-1))

if [[ -z "${models[$index]}" ]]; then
    echo "Invalid selection. Exiting."
    exit 1
fi

SELECTED_MODEL="${models[$index]}"
echo "Selected: $(basename "$SELECTED_MODEL")"

echo ""
read -p "Enter context length (default: $DEFAULT_CTX): " ctx_input

if [[ -z "$ctx_input" ]]; then
    CTX_LENGTH=$DEFAULT_CTX
else
    CTX_LENGTH=$ctx_input
fi

echo ""
echo "Starting llama-server on $HOST:$PORT..."
echo "Model: $SELECTED_MODEL"
echo "Context: $CTX_LENGTH"
echo "----------------------------------------"

llama-server \
    -m "$SELECTED_MODEL" \
    -c "$CTX_LENGTH" \
    --host "$HOST" \
    --port "$PORT" \
    --n-gpu-layers 99