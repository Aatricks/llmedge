#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
source "$ROOT_DIR/scripts/native_test_support.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/run_native_e2e.sh <suite>

Suites:
  text
  whisper
  video
  video-sequential
USAGE
}

suite="${1:-}"
if [[ -z "$suite" ]]; then
  usage
  exit 1
fi

run_gradle_test() {
  local test_filter="$1"
  local jvm_heap="$2"
  shift 2
  ./gradlew :llmedge:testDebugUnitTest \
    --tests "$test_filter" \
    --no-daemon \
    --console=plain \
    --warning-mode=none \
    "$@" \
    -Dorg.gradle.jvmargs="$jvm_heap"
}

ensure_native_env() {
  local target="$1"
  local native_dir
  local prebuilt_bin_dir
  local native_lib_name

  native_dir="$(llmedge_host_native_dir "$ROOT_DIR")"
  prebuilt_bin_dir="$(llmedge_prebuilt_bin_dir "$ROOT_DIR")"
  native_lib_name="$(llmedge_native_output_name "$target")"

  mkdir -p "$native_dir"
  llmedge_ensure_host_native_artifact "$ROOT_DIR" "$target" "$native_dir"

  export LLMEDGE_NATIVE_DIR="$native_dir"
  export LLMEDGE_PREBUILT_BIN_DIR="$prebuilt_bin_dir"
  export LLMEDGE_NATIVE_LIB_NAME="$native_lib_name"
}

run_text() {
  ensure_native_env smollm

  local models_dir="$ROOT_DIR/models"
  local default_model="$models_dir/SmolLM2-135M-Instruct-Q8_0.gguf"
  if [[ -z "${LLMEDGE_TEST_TEXT_MODEL_PATH:-}" ]]; then
    if [[ -f "$default_model" ]]; then
      echo "Found local model at $default_model"
      export LLMEDGE_TEST_TEXT_MODEL_PATH="$default_model"
    else
      echo "LLMEDGE_TEST_TEXT_MODEL_PATH is not set and $default_model not found."
      echo "Please run .github/scripts/download_models.sh or set the env var."
      exit 1
    fi
  fi

  export LD_LIBRARY_PATH="$LLMEDGE_NATIVE_DIR:${LD_LIBRARY_PATH:-}"
  export LLMEDGE_BUILD_NATIVE_LIB_PATH="$LLMEDGE_NATIVE_DIR/$LLMEDGE_NATIVE_LIB_NAME"

  echo "Running unit test: TextInferenceLinuxE2ETest"
  echo "LLMEDGE_TEST_TEXT_MODEL_PATH=$LLMEDGE_TEST_TEXT_MODEL_PATH"
  run_gradle_test \
    "io.aatricks.llmedge.TextInferenceLinuxE2ETest.desktop end-to-end text inference" \
    "-Xmx4g" \
    -DLLMEDGE_BUILD_NATIVE_LIB_PATH="$LLMEDGE_BUILD_NATIVE_LIB_PATH" \
    -DLLMEDGE_TEST_TEXT_MODEL_PATH="$LLMEDGE_TEST_TEXT_MODEL_PATH"
}

run_whisper() {
  ensure_native_env whisper

  if [[ -z "${LLMEDGE_TEST_WHISPER_MODEL_PATH:-}" ]]; then
    local models_dir="$ROOT_DIR/models"
    local model_file
    for model_file in "ggml-base.bin" "ggml-small.bin" "ggml-tiny.bin" "ggml-base.en.bin" "ggml-tiny.en.bin"; do
      if [[ -f "$models_dir/$model_file" ]]; then
        echo "Found local whisper model: $models_dir/$model_file"
        export LLMEDGE_TEST_WHISPER_MODEL_PATH="$models_dir/$model_file"
        break
      fi
    done

    if [[ -z "${LLMEDGE_TEST_WHISPER_MODEL_PATH:-}" ]]; then
      echo "LLMEDGE_TEST_WHISPER_MODEL_PATH is not set. Please set it to your whisper GGML model path."
      exit 1
    fi
  fi

  export LLMEDGE_BUILD_WHISPER_LIB_PATH="$LLMEDGE_NATIVE_DIR/$LLMEDGE_NATIVE_LIB_NAME"
  export LD_LIBRARY_PATH="$LLMEDGE_PREBUILT_BIN_DIR:$LLMEDGE_NATIVE_DIR:${LD_LIBRARY_PATH:-}"

  echo "Running unit test: WhisperLinuxE2ETest"
  echo "LLMEDGE_TEST_WHISPER_MODEL_PATH=${LLMEDGE_TEST_WHISPER_MODEL_PATH:-}"
  run_gradle_test \
    "*WhisperLinuxE2ETest" \
    "-Xmx4g" \
    -DLLMEDGE_BUILD_WHISPER_LIB_PATH="${LLMEDGE_BUILD_WHISPER_LIB_PATH:-}" \
    -DLLMEDGE_TEST_WHISPER_MODEL_PATH="${LLMEDGE_TEST_WHISPER_MODEL_PATH:-}" \
    --info
}

resolve_video_env_defaults() {
  local models_dir="$ROOT_DIR/models"
  if [[ -z "${LLMEDGE_TEST_MODEL_PATH:-}" && -z "${LLMEDGE_TEST_MODEL_ID:-}" ]]; then
    if [[ -f "$models_dir/wan2.1_t2v_1.3B_fp16.safetensors" ]]; then
      echo "Found local models in $models_dir"
      export LLMEDGE_TEST_MODEL_PATH="$models_dir/wan2.1_t2v_1.3B_fp16.safetensors"
      export LLMEDGE_TEST_VAE_PATH="$models_dir/wan_2.1_vae.safetensors"
      export LLMEDGE_TEST_T5_PATH="$models_dir/umt5-xxl-encoder-Q3_K_S.gguf"
    else
      echo "LLMEDGE_TEST_MODEL_PATH or LLMEDGE_TEST_MODEL_ID is not set. Please set one before running."
      exit 1
    fi
  fi

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
}

run_video() {
  ensure_native_env sdcpp
  resolve_video_env_defaults

  export LLMEDGE_BUILD_NATIVE_LIB_PATH="$LLMEDGE_NATIVE_DIR/$LLMEDGE_NATIVE_LIB_NAME"
  export LD_LIBRARY_PATH="$LLMEDGE_PREBUILT_BIN_DIR:$LLMEDGE_NATIVE_DIR:${LD_LIBRARY_PATH:-}"

  echo "Running unit test: VideoGenerationLinuxE2ETest"
  run_gradle_test \
    "io.aatricks.llmedge.VideoGenerationLinuxE2ETest.desktop end-to-end video generation" \
    "-Xmx8g" \
    -DLLMEDGE_BUILD_NATIVE_LIB_PATH="${LLMEDGE_BUILD_NATIVE_LIB_PATH:-}" \
    -DLLMEDGE_TEST_MODEL_ID="${LLMEDGE_TEST_MODEL_ID:-}" \
    -DLLMEDGE_TEST_MODEL_PATH="${LLMEDGE_TEST_MODEL_PATH:-}" \
    -DLLMEDGE_TEST_T5_PATH="${LLMEDGE_TEST_T5_PATH:-}" \
    -DLLMEDGE_TEST_VAE_PATH="${LLMEDGE_TEST_VAE_PATH:-}" \
    -DLLMEDGE_TEST_TAESD_PATH="${LLMEDGE_TEST_TAESD_PATH:-}" \
    -DHUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-}"
}

run_video_sequential() {
  ensure_native_env sdcpp

  local models_dir="$ROOT_DIR/models"
  export LLMEDGE_TEST_HF_CACHE_DIR="${LLMEDGE_TEST_HF_CACHE_DIR:-$models_dir/hf-models}"
  mkdir -p "$LLMEDGE_TEST_HF_CACHE_DIR"

  resolve_video_env_defaults

  export LLMEDGE_BUILD_NATIVE_LIB_PATH="$LLMEDGE_NATIVE_DIR/$LLMEDGE_NATIVE_LIB_NAME"
  export LD_LIBRARY_PATH="$LLMEDGE_PREBUILT_BIN_DIR:$LLMEDGE_NATIVE_DIR:${LD_LIBRARY_PATH:-}"

  echo "Running unit test: VideoGenerationSequentialE2ETest"
  run_gradle_test \
    "*VideoGenerationSequentialE2ETest" \
    "-Xmx12g" \
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
    -DLLMEDGE_TEST_ENABLE_I2V="${LLMEDGE_TEST_ENABLE_I2V:-false}"
}

case "$suite" in
  text)
    run_text
    ;;
  whisper)
    run_whisper
    ;;
  video)
    run_video
    ;;
  video-sequential)
    run_video_sequential
    ;;
  *)
    usage
    exit 1
    ;;
esac

echo "Done."
