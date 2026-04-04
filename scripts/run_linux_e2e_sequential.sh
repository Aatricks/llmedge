#!/usr/bin/env bash
set -euo pipefail

# Script to run the sequential video generation E2E test that matches the Android device path.
# This test uses forceSequentialLoad=true and txt2VidWithPrecomputedCondition - the exact
# same code path that Android devices use due to memory constraints.

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
source "$ROOT_DIR/scripts/native_test_support.sh"
LLMEDGE_NATIVE_DIR="$(llmedge_host_native_dir "$ROOT_DIR")"
NATIVE_LIB_NAME="$(llmedge_native_output_name sdcpp)"
PREBUILT_BIN_DIR="$(llmedge_prebuilt_bin_dir "$ROOT_DIR")"

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
llmedge_ensure_host_native_artifact "$ROOT_DIR" sdcpp "$LLMEDGE_NATIVE_DIR"

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
