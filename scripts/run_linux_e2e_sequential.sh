#!/usr/bin/env bash
set -euo pipefail

# Script to run the sequential video generation E2E test that matches the Android device path.
# This test uses forceSequentialLoad=true and txt2VidWithPrecomputedCondition - the exact
# same code path that Android devices use due to memory constraints.

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
LLMEDGE_NATIVE_DIR="$ROOT_DIR/llmedge/build/native/linux-x86_64"
NATIVE_LIB_NAME="libsdcpp.so"
DEP_LIB_NAME="libstable-diffusion.so"
PREBUILT_BIN_DIR="$ROOT_DIR/scripts/jni-desktop/build/bin"

echo "============================================"
echo "Sequential Video Generation E2E Test"
echo "(Android Device Path Simulation)"
echo "============================================"

# Check for model files OR allow HF mode.
MODELS_DIR="$ROOT_DIR/models"

# Default persistent cache for HF downloads (avoids re-downloading across Robolectric runs)
if [[ -z "${LLMEDGE_TEST_HF_CACHE_DIR:-}" ]]; then
  export LLMEDGE_TEST_HF_CACHE_DIR="$MODELS_DIR/hf-models"
fi
mkdir -p "$LLMEDGE_TEST_HF_CACHE_DIR"

if [[ -z "${LLMEDGE_TEST_MODEL_PATH:-}" ]]; then
  if [[ -f "$MODELS_DIR/wan2.1_t2v_1.3B_fp16.safetensors" ]]; then
    echo "Found local models in $MODELS_DIR"
    export LLMEDGE_TEST_MODEL_PATH="$MODELS_DIR/wan2.1_t2v_1.3B_fp16.safetensors"
    export LLMEDGE_TEST_VAE_PATH="$MODELS_DIR/wan_2.1_vae.safetensors"
    export LLMEDGE_TEST_T5_PATH="$MODELS_DIR/umt5-xxl-encoder-Q3_K_S.gguf"
  elif [[ -n "${LLMEDGE_TEST_MODEL_ID:-}" ]]; then
    echo "No local model paths set; using Hugging Face download mode (LLMEDGE_TEST_MODEL_ID=${LLMEDGE_TEST_MODEL_ID})."
    echo "Downloads will be cached under: $LLMEDGE_TEST_HF_CACHE_DIR"
  else
    echo "ERROR: Model files not found!"
    echo ""
    echo "Please do one of the following:"
    echo "  1. Place model files in $MODELS_DIR:"
    echo "     - wan2.1_t2v_1.3B_fp16.safetensors"
    echo "     - wan_2.1_vae.safetensors"
    echo "     - umt5-xxl-encoder-Q3_K_S.gguf"
    echo ""
    echo "  2. Or set environment variables (paths mode):"
    echo "     export LLMEDGE_TEST_MODEL_PATH=/path/to/wan2.1_t2v_1.3B_fp16.safetensors"
    echo "     export LLMEDGE_TEST_VAE_PATH=/path/to/wan_2.1_vae.safetensors"
    echo "     export LLMEDGE_TEST_T5_PATH=/path/to/umt5-xxl-encoder-Q3_K_S.gguf"
    echo ""
    echo "  3. Or set environment variables (HF mode):"
    echo "     export LLMEDGE_TEST_MODEL_ID=Comfy-Org/Wan_2.1_ComfyUI_repackaged"
    echo "     export LLMEDGE_TEST_MODEL_FILENAME=wan2.1_t2v_1.3B_fp16.safetensors"
    echo "     export LLMEDGE_TEST_VAE_FILENAME=wan_2.1_vae.safetensors"
    echo "     export LLMEDGE_TEST_T5_MODEL_ID=city96/umt5-xxl-encoder-gguf"
    echo "     export LLMEDGE_TEST_T5_FILENAME=umt5-xxl-encoder-Q3_K_S.gguf"
    echo ""
    echo "(Optional) export LLMEDGE_TEST_HF_CACHE_DIR=$MODELS_DIR/hf-models"
    exit 1
  fi
fi

echo ""
echo "Model paths:"
echo "  MODEL: ${LLMEDGE_TEST_MODEL_PATH:-NOT SET}"
echo "  VAE:   ${LLMEDGE_TEST_VAE_PATH:-NOT SET}"
echo "  T5:    ${LLMEDGE_TEST_T5_PATH:-NOT SET}"
echo "  HF CACHE: ${LLMEDGE_TEST_HF_CACHE_DIR:-NOT SET}"
echo "  ENABLE I2V: ${LLMEDGE_TEST_ENABLE_I2V:-false}"
echo ""

# Ensure native library directory exists
mkdir -p "$LLMEDGE_NATIVE_DIR"

# Check/build native library
if [[ -f "$LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME" ]]; then
  echo "Found native library at $LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME"
elif [[ -d "$PREBUILT_BIN_DIR" && -f "$PREBUILT_BIN_DIR/$NATIVE_LIB_NAME" ]]; then
  echo "Copying prebuilt $NATIVE_LIB_NAME from $PREBUILT_BIN_DIR"
  cp "$PREBUILT_BIN_DIR/$NATIVE_LIB_NAME" "$LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME"
  if [[ -f "$PREBUILT_BIN_DIR/$DEP_LIB_NAME" ]]; then
    cp "$PREBUILT_BIN_DIR/$DEP_LIB_NAME" "$LLMEDGE_NATIVE_DIR/$DEP_LIB_NAME"
  fi
else
  echo "Native library not found. Building with scripts/build_native_linux.sh sdcpp..."
  if [[ -f "$ROOT_DIR/scripts/build_native_linux.sh" ]]; then
    "$ROOT_DIR/scripts/build_native_linux.sh" sdcpp
  else
    echo "ERROR: No build script found. Please build libsdcpp for host."
    exit 1
  fi
fi

# Set up environment
export LLMEDGE_BUILD_NATIVE_LIB_PATH="$LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME"
export LD_LIBRARY_PATH="$PREBUILT_BIN_DIR:$LLMEDGE_NATIVE_DIR:${LD_LIBRARY_PATH:-}"

echo ""
echo "Running test: VideoGenerationSequentialE2ETest"
echo "This test simulates the exact Android device code path:"
echo "  1. Load T5 encoder -> precompute conditions -> unload T5"
echo "  2. Load diffusion model + VAE -> txt2VidWithPrecomputedCondition"
echo ""

./gradlew :llmedge:testDebugUnitTest \
  --tests "*VideoGenerationSequentialE2ETest" \
  --no-daemon \
  --console=plain \
  --info \
  -DLLMEDGE_BUILD_NATIVE_LIB_PATH="${LLMEDGE_BUILD_NATIVE_LIB_PATH:-}" \
  -DLLMEDGE_TEST_MODEL_PATH="${LLMEDGE_TEST_MODEL_PATH:-}" \
  -DLLMEDGE_TEST_T5_PATH="${LLMEDGE_TEST_T5_PATH:-}" \
  -DLLMEDGE_TEST_VAE_PATH="${LLMEDGE_TEST_VAE_PATH:-}" \
  -DLLMEDGE_TEST_MODEL_ID="${LLMEDGE_TEST_MODEL_ID:-}" \
  -DLLMEDGE_TEST_MODEL_FILENAME="${LLMEDGE_TEST_MODEL_FILENAME:-}" \
  -DLLMEDGE_TEST_VAE_FILENAME="${LLMEDGE_TEST_VAE_FILENAME:-}" \
  -DLLMEDGE_TEST_T5_MODEL_ID="${LLMEDGE_TEST_T5_MODEL_ID:-}" \
  -DLLMEDGE_TEST_T5_FILENAME="${LLMEDGE_TEST_T5_FILENAME:-}" \
  -DLLMEDGE_TEST_HF_CACHE_DIR="${LLMEDGE_TEST_HF_CACHE_DIR:-}" \
  -DLLMEDGE_TEST_ENABLE_I2V="${LLMEDGE_TEST_ENABLE_I2V:-false}" \
  -Dorg.gradle.jvmargs="-Xmx12g"

echo ""
echo "============================================"
echo "Test completed!"
echo "============================================"
