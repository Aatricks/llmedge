#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP_CMAKE_DIR="$ROOT_DIR/scripts/jni-desktop"
BUILD_BIN_DIR="$DESKTOP_CMAKE_DIR/build/bin"

usage() {
    cat <<'EOF'
Usage: scripts/build_native_linux.sh <target> [<target> ...]

Targets:
  smollm
  whisper
  sdcpp
  bark
  all
EOF
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

ARCH=$(uname -m)
case "$ARCH" in
  x86_64) ARCH_DIR="linux-x86_64" ;;
  aarch64|arm64) ARCH_DIR="linux-aarch64" ;;
  *) ARCH_DIR="linux-$ARCH" ;;
esac

copy_output() {
    local built_lib="$1"
    local target_name="$2"
    local output_name="$3"

    mkdir -p "$ROOT_DIR/llmedge/build/native/$ARCH_DIR"
    cp "$built_lib" "$ROOT_DIR/llmedge/build/native/$ARCH_DIR/$output_name"

    mkdir -p "$BUILD_BIN_DIR"
    cp "$built_lib" "$BUILD_BIN_DIR/$output_name"
}

prepare_sdcpp_mods() {
    local build_dir="$1"
    local -n out_args="$2"

    local use_mods="${LLMEDGE_SDCPP_USE_MODS:-}"
    if [[ -z "$use_mods" ]]; then
      if [[ -f "$ROOT_DIR/mods/stable-diffusion.cpp" || -f "$ROOT_DIR/mods/stable-diffusion.h" || -f "$ROOT_DIR/mods/wan.hpp" ]]; then
        use_mods=1
      else
        use_mods=0
      fi
    fi

    if [[ "$use_mods" != "1" ]]; then
        return
    fi

    local patched_root="$build_dir/patched-sd-src"
    rm -rf "$patched_root"
    mkdir -p "$patched_root"
    cp -a "$ROOT_DIR/stable-diffusion.cpp/." "$patched_root/"

    local default_mods="stable-diffusion.cpp,stable-diffusion.h,ggml_extend.hpp,util.cpp,ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp"
    if [[ -f "$ROOT_DIR/mods/wan.hpp" ]]; then
      default_mods="wan.hpp,$default_mods"
    fi
    local mods_raw="${LLMEDGE_SDCPP_MODS_FILES:-$default_mods}"
    IFS=',' read -r -a mods_files <<< "$mods_raw"
    for f in "${mods_files[@]}"; do
      local trimmed="${f//[[:space:]]/}"
      [[ -z "$trimmed" ]] && continue
      local target_path="$trimmed"
      if [[ "$trimmed" == "stable-diffusion.h" ]]; then
        target_path="include/stable-diffusion.h"
      elif [[ "$trimmed" == *.h ]]; then
        if [[ -f "$patched_root/include/$trimmed" ]]; then
          target_path="include/$trimmed"
        else
          target_path="src/$trimmed"
        fi
      elif [[ "$trimmed" == *.cpp ]] || [[ "$trimmed" == *.hpp ]] || [[ "$trimmed" == *.inl ]]; then
        target_path="src/$trimmed"
      fi

      if [[ -f "$ROOT_DIR/mods/$trimmed" ]]; then
        mkdir -p "$(dirname "$patched_root/$target_path")"
        cp -a "$ROOT_DIR/mods/$trimmed" "$patched_root/$target_path"
      fi
    done

    out_args+=("-DSD_ROOT_OVERRIDE=$patched_root")
}

build_sdcpp() {
    local build_dir="$DESKTOP_CMAKE_DIR/build-sdcpp"
    mkdir -p "$build_dir"
    local cmake_args=()
    prepare_sdcpp_mods "$build_dir" cmake_args

    cmake -S "$DESKTOP_CMAKE_DIR" -B "$build_dir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SDCPP=ON \
      -DBUILD_SMOLLM=OFF \
      -DBUILD_BARK=OFF \
      -DWHISPER_DESKTOP_JNI=OFF \
      -DSD_VULKAN=ON \
      -DWAN_SUPPORT=ON \
      -DSPDLOG_FMT_EXTERNAL=ON \
      -DGGML_SKIP_OSX_FEATURES=ON \
      -DSDC_TEST_DESKTOP_JNI=ON \
      "${cmake_args[@]}"

    cmake --build "$build_dir" --target sdcpp --parallel "$(nproc)"
    local lib_path
    lib_path=$(find "$build_dir" -type f -name 'libsdcpp*.so' -print -quit || true)
    if [[ -z "$lib_path" ]]; then
        echo "libsdcpp.so not found under $build_dir" >&2
        exit 1
    fi
    copy_output "$lib_path" "sdcpp" "libsdcpp.so"
}

build_smollm() {
    local build_dir="$DESKTOP_CMAKE_DIR/build-smollm"
    mkdir -p "$build_dir"

    cmake -S "$DESKTOP_CMAKE_DIR" -B "$build_dir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SDCPP=OFF \
      -DBUILD_SMOLLM=ON \
      -DBUILD_BARK=OFF \
      -DWHISPER_DESKTOP_JNI=OFF \
      -DGGML_VULKAN=OFF \
      -DSPDLOG_FMT_EXTERNAL=ON \
      -DGGML_SKIP_OSX_FEATURES=ON

    cmake --build "$build_dir" --target smollm --parallel "$(nproc)"
    local lib_path
    lib_path=$(find "$build_dir" -type f -name 'libsmollm*.so' -print -quit || true)
    if [[ -z "$lib_path" ]]; then
        echo "libsmollm.so not found under $build_dir" >&2
        exit 1
    fi
    copy_output "$lib_path" "smollm" "libsmollm.so"
    cp "$ROOT_DIR/llmedge/build/native/$ARCH_DIR/libsmollm.so" "$ROOT_DIR/llmedge/build/native/$ARCH_DIR/libsmollm_v7a.so"
    cp "$ROOT_DIR/llmedge/build/native/$ARCH_DIR/libsmollm.so" "$ROOT_DIR/llmedge/build/native/$ARCH_DIR/libsmollm_v8.so"
}

build_whisper() {
    local build_dir="$DESKTOP_CMAKE_DIR/build-whisper"
    mkdir -p "$build_dir"

    cmake -S "$DESKTOP_CMAKE_DIR" -B "$build_dir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SDCPP=OFF \
      -DBUILD_SMOLLM=OFF \
      -DBUILD_BARK=OFF \
      -DWHISPER_DESKTOP_JNI=ON

    cmake --build "$build_dir" --target whisper_jni --parallel "$(nproc)"
    copy_output "$build_dir/libwhisper_jni.so" "whisper" "libwhisper_jni.so"
}

build_bark() {
    local build_dir="$DESKTOP_CMAKE_DIR/build-bark"
    mkdir -p "$build_dir"

    cmake -S "$DESKTOP_CMAKE_DIR" -B "$build_dir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SDCPP=OFF \
      -DBUILD_SMOLLM=OFF \
      -DBUILD_BARK=ON \
      -DWHISPER_DESKTOP_JNI=OFF

    cmake --build "$build_dir" --target bark_jni --parallel "$(nproc)"
    copy_output "$build_dir/libbark_jni.so" "bark" "libbark_jni.so"
}

targets=("$@")
if [[ " ${targets[*]} " == *" all "* ]]; then
    targets=(smollm whisper sdcpp bark)
fi

for target in "${targets[@]}"; do
    case "$target" in
      smollm) build_smollm ;;
      whisper) build_whisper ;;
      sdcpp) build_sdcpp ;;
      bark) build_bark ;;
      *)
        echo "Unknown target: $target" >&2
        usage
        exit 1
        ;;
    esac
done
