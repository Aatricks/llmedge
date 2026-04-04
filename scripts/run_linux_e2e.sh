#!/usr/bin/env bash
set -euo pipefail

# Helper script to run the headless Linux E2E video generation test.
# It will try to locate or build the host native libsdcpp.so and then run the
# Robolectric DebugUnit test that uses the native library and a GGUF model.

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
LLMEDGE_NATIVE_DIR="$ROOT_DIR/llmedge/build/native/linux-x86_64"
NATIVE_LIB_NAME="libsdcpp.so"
DEP_LIB_NAME="libstable-diffusion.so"
PREBUILT_BIN_DIR="$ROOT_DIR/scripts/jni-desktop/build/bin"

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

# If the native lib already exists in the target folder, use it.
if [[ -f "$LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME" ]]; then
  echo "Found native library at $LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME"
else
  # Try to copy from prebuilt jni-desktop build outputs
  if [[ -d "$PREBUILT_BIN_DIR" && -f "$PREBUILT_BIN_DIR/$NATIVE_LIB_NAME" ]]; then
    echo "Copying prebuilt $NATIVE_LIB_NAME from $PREBUILT_BIN_DIR"
    cp "$PREBUILT_BIN_DIR/$NATIVE_LIB_NAME" "$LLMEDGE_NATIVE_DIR/$NATIVE_LIB_NAME"
    # If jni-desktop also produced libstable-diffusion, copy it next to the native lib so the dynamic
    # loader can resolve transitive dependencies when System.load is called from the JVM.
    if [[ -f "$PREBUILT_BIN_DIR/$DEP_LIB_NAME" ]]; then
      echo "Copying dependent $DEP_LIB_NAME from $PREBUILT_BIN_DIR to $LLMEDGE_NATIVE_DIR"
      cp "$PREBUILT_BIN_DIR/$DEP_LIB_NAME" "$LLMEDGE_NATIVE_DIR/$DEP_LIB_NAME"
    fi
  else
    echo "Prebuilt libs not found. Attempting to build with scripts/build_native_linux.sh sdcpp"
    if [[ -f "$ROOT_DIR/scripts/build_native_linux.sh" ]]; then
      "$ROOT_DIR/scripts/build_native_linux.sh" sdcpp
    else
      echo "No build script found; please build libsdcpp for host and place it in $LLMEDGE_NATIVE_DIR"
      exit 1
    fi
  fi
fi

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
