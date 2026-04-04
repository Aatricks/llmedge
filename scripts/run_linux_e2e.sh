#!/usr/bin/env bash
set -euo pipefail

# Helper script to run the headless Linux E2E video generation test.
# It will try to locate or build the host native libsdcpp.so and then run the
# Robolectric DebugUnit test that uses the native library and a GGUF model.

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
source "$ROOT_DIR/scripts/native_test_support.sh"
LLMEDGE_NATIVE_DIR="$(llmedge_host_native_dir "$ROOT_DIR")"
NATIVE_LIB_NAME="$(llmedge_native_output_name sdcpp)"
PREBUILT_BIN_DIR="$(llmedge_prebuilt_bin_dir "$ROOT_DIR")"

if [[ -z "${LLMEDGE_TEST_MODEL_PATH:-}" && -z "${LLMEDGE_TEST_MODEL_ID:-}" ]]; then
  # Check for local models in models/ directory
  MODELS_DIR="$ROOT_DIR/models"
  if [[ -f "$MODELS_DIR/wan2.1_t2v_1.3B_fp16.safetensors" ]]; then
    echo "Found local models in $MODELS_DIR"
    export LLMEDGE_TEST_MODEL_PATH="$MODELS_DIR/wan2.1_t2v_1.3B_fp16.safetensors"
    export LLMEDGE_TEST_VAE_PATH="$MODELS_DIR/wan_2.1_vae.safetensors"
    export LLMEDGE_TEST_T5_PATH="$MODELS_DIR/umt5-xxl-encoder-Q3_K_S.gguf"
  else
    echo "LLMEDGE_TEST_MODEL_PATH or LLMEDGE_TEST_MODEL_ID is not set. Please set one before running."
    echo "For the Wan2.1 + umt5 pair, set either LLMEDGE_TEST_MODEL_PATH to your Wan GGUF path or set LLMEDGE_TEST_MODEL_ID=wan/Wan2.1-T2V-1.3B and LLMEDGE_TEST_T5_PATH to your T5 gguf path."
    exit 1
  fi
fi

mkdir -p "$LLMEDGE_NATIVE_DIR"
llmedge_ensure_host_native_artifact "$ROOT_DIR" sdcpp "$LLMEDGE_NATIVE_DIR"

echo "Running unit test: VideoGenerationLinuxE2ETest"
echo "LLMEDGE_TEST_MODEL_ID=${LLMEDGE_TEST_MODEL_ID:-}"
echo "LLMEDGE_TEST_MODEL_PATH=${LLMEDGE_TEST_MODEL_PATH:-}"
if [[ -n "${LLMEDGE_TEST_MODEL_PATH:-}" ]]; then
  export LLMEDGE_TEST_MODEL_PATH="$LLMEDGE_TEST_MODEL_PATH"
fi
if [[ -n "${LLMEDGE_TEST_MODEL_ID:-}" ]]; then
  export LLMEDGE_TEST_MODEL_ID="$LLMEDGE_TEST_MODEL_ID"
fi
if [[ -n "${LLMEDGE_TEST_T5_PATH:-}" ]]; then
  export LLMEDGE_TEST_T5_PATH="$LLMEDGE_TEST_T5_PATH"
fi
if [[ -n "${LLMEDGE_TEST_VAE_PATH:-}" ]]; then
  export LLMEDGE_TEST_VAE_PATH="$LLMEDGE_TEST_VAE_PATH"
fi
if [[ -n "${LLMEDGE_TEST_TAESD_PATH:-}" ]]; then
  export LLMEDGE_TEST_TAESD_PATH="$LLMEDGE_TEST_TAESD_PATH"
fi
export LLMEDGE_BUILD_NATIVE_LIB_PATH="$LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME"
echo "LLMEDGE_BUILD_NATIVE_LIB_PATH=${LLMEDGE_BUILD_NATIVE_LIB_PATH:-}"
echo "LLMEDGE_TEST_T5_PATH=${LLMEDGE_TEST_T5_PATH:-}"
echo "LLMEDGE_TEST_VAE_PATH=${LLMEDGE_TEST_VAE_PATH:-}"
echo "LLMEDGE_TEST_TAESD_PATH=${LLMEDGE_TEST_TAESD_PATH:-}"

# Ensure the dynamic linker can resolve dependent shared libraries when loading libsdcpp.so.
# Prefer adding the prebuilt bin dir first, then native dir, so that transitive shared libs are found.
export LD_LIBRARY_PATH="$PREBUILT_BIN_DIR:$LLMEDGE_NATIVE_DIR:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "Environment variables visible to the process:"
env | grep -i llmedge || true

./gradlew :llmedge:testDebugUnitTest \
  --tests "io.aatricks.llmedge.VideoGenerationLinuxE2ETest.desktop end-to-end video generation" \
  --no-daemon \
  --console=plain \
  --warning-mode=none \
  -DLLMEDGE_BUILD_NATIVE_LIB_PATH="${LLMEDGE_BUILD_NATIVE_LIB_PATH:-}" \
  -DLLMEDGE_TEST_MODEL_ID="${LLMEDGE_TEST_MODEL_ID:-}" \
  -DLLMEDGE_TEST_MODEL_PATH="${LLMEDGE_TEST_MODEL_PATH:-}" \
  -DLLMEDGE_TEST_T5_PATH="${LLMEDGE_TEST_T5_PATH:-}" \
  -DLLMEDGE_TEST_VAE_PATH="${LLMEDGE_TEST_VAE_PATH:-}" \
  -DLLMEDGE_TEST_TAESD_PATH="${LLMEDGE_TEST_TAESD_PATH:-}" \
  -DHUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-}" \
  -Dorg.gradle.jvmargs="-Xmx8g"

echo "Done."
