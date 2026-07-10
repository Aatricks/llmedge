#!/usr/bin/env bash
set -e

# Download models for E2E tests if space permits

mkdir -p models
cd models

# Check available disk space
AVAILABLE_SPACE_KB=$(df -k . | awk 'NR==2 {print $4}')
echo "Available space: $((AVAILABLE_SPACE_KB / 1024 / 1024)) GB"

# 1. Audio Model: Whisper Tiny (Small, ~75MB)
echo "Downloading Whisper Tiny model..."
if [ ! -f "ggml-tiny.en.bin" ]; then
    curl -L -o ggml-tiny.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
fi
echo "Whisper model downloaded."

# 2. Text Model: SmolLM2 135M (Small, ~150MB)
# We need a GGUF model for SmolLM/Llama.cpp
echo "Downloading SmolLM2 model..."
# Cleanup old failed download if exists
rm -f "SmolLM-135M-Instruct-q8_0.gguf"

if [ ! -f "SmolLM2-135M-Instruct-Q8_0.gguf" ]; then
    curl -L -o SmolLM2-135M-Instruct-Q8_0.gguf https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf
fi
echo "SmolLM2 model downloaded."

# 3. Image Model: SD-Turbo (~1.3GB) — only if there is comfortable headroom
if [ "$AVAILABLE_SPACE_KB" -gt $((8 * 1024 * 1024)) ]; then
    echo "Downloading SD-Turbo model..."
    if [ ! -f "sd_turbo-f16-q8_0.gguf" ]; then
        curl -L -o sd_turbo-f16-q8_0.gguf https://huggingface.co/Green-Sky/SD-Turbo-GGUF/resolve/main/sd_turbo-f16-q8_0.gguf
    fi
    echo "SD-Turbo model downloaded."
else
    echo "Skipping SD-Turbo download (insufficient disk space); image E2E will be skipped."
fi

