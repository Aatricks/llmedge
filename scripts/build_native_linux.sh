#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP_CMAKE_DIR="$ROOT_DIR/scripts/jni-desktop"
BUILD_BIN_DIR="$DESKTOP_CMAKE_DIR/build/bin"
source "$ROOT_DIR/scripts/native_targets.sh"

usage() {
    cat <<'EOF'
Usage: scripts/build_native_linux.sh <target> [<target> ...]

Targets:
EOF
    llmedge_print_native_targets
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
    local output_name="$2"

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

configure_target() {
    local target="$1"
    local build_dir="$2"
    local -n out_args="$3"

    out_args=(-DCMAKE_BUILD_TYPE=Release)
    case "$target" in
        sdcpp)
            prepare_sdcpp_mods "$build_dir" out_args
            out_args+=(
                -DBUILD_SDCPP=ON
                -DBUILD_SMOLLM=OFF
                -DBUILD_BARK=OFF
                -DWHISPER_DESKTOP_JNI=OFF
                -DSD_VULKAN=ON
                -DWAN_SUPPORT=ON
                -DSPDLOG_FMT_EXTERNAL=ON
                -DGGML_SKIP_OSX_FEATURES=ON
                -DSDC_TEST_DESKTOP_JNI=ON
            )
            ;;
        smollm)
            out_args+=(
                -DBUILD_SDCPP=OFF
                -DBUILD_SMOLLM=ON
                -DBUILD_BARK=OFF
                -DWHISPER_DESKTOP_JNI=OFF
                -DGGML_VULKAN=OFF
                -DSPDLOG_FMT_EXTERNAL=ON
                -DGGML_SKIP_OSX_FEATURES=ON
            )
            ;;
        whisper)
            out_args+=(
                -DBUILD_SDCPP=OFF
                -DBUILD_SMOLLM=OFF
                -DBUILD_BARK=OFF
                -DWHISPER_DESKTOP_JNI=ON
            )
            ;;
        bark)
            out_args+=(
                -DBUILD_SDCPP=OFF
                -DBUILD_SMOLLM=OFF
                -DBUILD_BARK=ON
                -DWHISPER_DESKTOP_JNI=OFF
            )
            ;;
        *)
            echo "Unknown target: $target" >&2
            return 1
            ;;
    esac
}

copy_alias_outputs() {
    local target="$1"
    local output_name="$2"
    local alias_name
    while IFS= read -r alias_name; do
        [[ -n "$alias_name" ]] || continue
        cp "$ROOT_DIR/llmedge/build/native/$ARCH_DIR/$output_name" \
            "$ROOT_DIR/llmedge/build/native/$ARCH_DIR/$alias_name"
    done < <(llmedge_native_alias_outputs "$target")
}

build_target() {
    local target="$1"
    local build_dir="$DESKTOP_CMAKE_DIR/$(llmedge_native_build_dir_name "$target")"
    local cmake_target
    local output_name
    local lib_path
    local cmake_args=()

    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    configure_target "$target" "$build_dir" cmake_args

    cmake -S "$DESKTOP_CMAKE_DIR" -B "$build_dir" "${cmake_args[@]}"

    cmake_target="$(llmedge_native_cmake_target "$target")"
    cmake --build "$build_dir" --target "$cmake_target" --parallel "$(nproc)"

    output_name="$(llmedge_native_output_name "$target")"
    lib_path=$(find "$build_dir" -type f -name "$output_name" -print -quit || true)
    if [[ -z "$lib_path" ]]; then
        echo "$output_name not found under $build_dir" >&2
        exit 1
    fi
    copy_output "$lib_path" "$output_name"
    copy_alias_outputs "$target" "$output_name"
}

targets=("$@")
if [[ " ${targets[*]} " == *" all "* ]]; then
    targets=("${LLMEDGE_DESKTOP_TARGETS[@]}")
fi

for target in "${targets[@]}"; do
    if ! llmedge_is_known_native_target "$target"; then
        echo "Unknown target: $target" >&2
        usage
        exit 1
    fi
    build_target "$target"
done
