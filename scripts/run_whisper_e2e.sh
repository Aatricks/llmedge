#!/usr/bin/env bash
set -euo pipefail

# Helper script to run the headless Linux E2E Whisper transcription test.
# It will try to locate or build the host native libwhisper_jni.so and then run the
# Robolectric test that uses the native library and a GGML model.

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
source "$ROOT_DIR/scripts/native_test_support.sh"
LLMEDGE_NATIVE_DIR="$(llmedge_host_native_dir "$ROOT_DIR")"
NATIVE_LIB_NAME="$(llmedge_native_output_name whisper)"
PREBUILT_BIN_DIR="$(llmedge_prebuilt_bin_dir "$ROOT_DIR")"

if [[ -z "${LLMEDGE_TEST_WHISPER_MODEL_PATH:-}" ]]; then
    # Check for local models in models/ directory
    MODELS_DIR="$ROOT_DIR/models"

    # Look for whisper models with common names
    for model_file in "ggml-base.bin" "ggml-small.bin" "ggml-tiny.bin" "ggml-base.en.bin" "ggml-tiny.en.bin"; do
        if [[ -f "$MODELS_DIR/$model_file" ]]; then
            echo "Found local whisper model: $MODELS_DIR/$model_file"
            export LLMEDGE_TEST_WHISPER_MODEL_PATH="$MODELS_DIR/$model_file"
            break
        fi
    done

    if [[ -z "${LLMEDGE_TEST_WHISPER_MODEL_PATH:-}" ]]; then
        echo "LLMEDGE_TEST_WHISPER_MODEL_PATH is not set. Please set it to your whisper GGML model path."
        echo "You can download a model from: https://huggingface.co/ggerganov/whisper.cpp"
        echo "Example: wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin -O models/ggml-base.bin"
        exit 1
    fi
fi

mkdir -p "$LLMEDGE_NATIVE_DIR"
llmedge_ensure_host_native_artifact "$ROOT_DIR" whisper "$LLMEDGE_NATIVE_DIR"

echo "Running unit test: WhisperLinuxE2ETest"
echo "LLMEDGE_TEST_WHISPER_MODEL_PATH=${LLMEDGE_TEST_WHISPER_MODEL_PATH:-}"

export LLMEDGE_BUILD_WHISPER_LIB_PATH="$LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME"
echo "LLMEDGE_BUILD_WHISPER_LIB_PATH=${LLMEDGE_BUILD_WHISPER_LIB_PATH:-}"

# Ensure the dynamic linker can resolve dependent shared libraries
export LD_LIBRARY_PATH="$PREBUILT_BIN_DIR:$LLMEDGE_NATIVE_DIR:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "Environment variables visible to the process:"
env | grep -i llmedge || true

./gradlew :llmedge:testDebugUnitTest \
    --tests "*WhisperLinuxE2ETest" \
    --no-daemon \
    --console=plain \
    --info \
    -DLLMEDGE_BUILD_WHISPER_LIB_PATH="${LLMEDGE_BUILD_WHISPER_LIB_PATH:-}" \
    -DLLMEDGE_TEST_WHISPER_MODEL_PATH="${LLMEDGE_TEST_WHISPER_MODEL_PATH:-}" \
    -Dorg.gradle.jvmargs="-Xmx4g"

echo "Done."
