#!/usr/bin/env bash
set -euo pipefail

# Helper script to run the headless Linux E2E text inference test.
# It will try to locate or build the host native libsmollm.so and then run the
# Robolectric DebugUnit test.

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
LLMEDGE_NATIVE_DIR="$ROOT_DIR/llmedge/build/native/linux-x86_64"
NATIVE_LIB_NAME="libsmollm.so"
MODELS_DIR="$ROOT_DIR/models"

# Default model path
DEFAULT_MODEL="$MODELS_DIR/SmolLM2-135M-Instruct-Q8_0.gguf"

if [[ -z "${LLMEDGE_TEST_TEXT_MODEL_PATH:-}" ]]; then
  if [[ -f "$DEFAULT_MODEL" ]]; then
    echo "Found local model at $DEFAULT_MODEL"
    export LLMEDGE_TEST_TEXT_MODEL_PATH="$DEFAULT_MODEL"
  else
    echo "LLMEDGE_TEST_TEXT_MODEL_PATH is not set and $DEFAULT_MODEL not found."
    echo "Please run .github/scripts/download_models.sh or set the env var."
    exit 1
  fi
fi

mkdir -p "$LLMEDGE_NATIVE_DIR"

# Check/Build native lib
if [[ -f "$LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME" ]]; then
  echo "Found native library at $LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME"
else
  echo "Native lib not found. Attempting to build with scripts/build_native_linux.sh smollm"
  if [[ -f "$ROOT_DIR/scripts/build_native_linux.sh" ]]; then
    "$ROOT_DIR/scripts/build_native_linux.sh" smollm
  else
    echo "No build script found."
    exit 1
  fi
fi

echo "Running unit test: TextInferenceLinuxE2ETest"
echo "LLMEDGE_TEST_TEXT_MODEL_PATH=$LLMEDGE_TEST_TEXT_MODEL_PATH"

# Ensure dynamic linker can resolve libsmollm.so
# Also add llama.cpp/common libs if they are shared? 
# Our CMake build compiles them into libsmollm.so (PRIVATE link) or static?
# If common/llama are static (default in our cmake), then libsmollm.so is self-contained regarding llama.
export LD_LIBRARY_PATH="$LLMEDGE_NATIVE_DIR:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

export LLMEDGE_BUILD_NATIVE_LIB_PATH="$LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME"
echo "LLMEDGE_BUILD_NATIVE_LIB_PATH=$LLMEDGE_BUILD_NATIVE_LIB_PATH"

./gradlew :llmedge:testDebugUnitTest \
  --tests "io.aatricks.llmedge.TextInferenceLinuxE2ETest.desktop end-to-end text inference" \
  --no-daemon \
  --console=plain \
  --warning-mode=none \
  -DLLMEDGE_BUILD_NATIVE_LIB_PATH="$LLMEDGE_BUILD_NATIVE_LIB_PATH" \
  -DLLMEDGE_TEST_TEXT_MODEL_PATH="$LLMEDGE_TEST_TEXT_MODEL_PATH" \
  -Dorg.gradle.jvmargs="-Xmx4g"

echo "Done."
