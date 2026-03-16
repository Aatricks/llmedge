#!/usr/bin/env bash
set -euo pipefail

# Build sdcpp JNI library for Linux (x86_64). Places the resulting libsdcpp.so into build/native/linux-x86_64

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
BUILD_DIR="$ROOT_DIR/scripts/jni-desktop/build"
mkdir -p "$BUILD_DIR"

# Optional: build against a patched copy of stable-diffusion.cpp by overlaying the
# contents of the repo's `mods/` directory.
#
# This is useful for keeping the `stable-diffusion.cpp` submodule clean (no `-dirty`)
# while still compiling llmedge-specific extensions.
USE_MODS="${LLMEDGE_SDCPP_USE_MODS:-}"
if [[ -z "${USE_MODS}" ]]; then
  if [[ -f "$ROOT_DIR/mods/stable-diffusion.cpp" || -f "$ROOT_DIR/mods/stable-diffusion.h" || -f "$ROOT_DIR/mods/wan.hpp" ]]; then
    USE_MODS=1
  else
    USE_MODS=0
  fi
fi
CMAKE_EXTRA_ARGS=()
if [[ "$USE_MODS" == "1" ]]; then
  PATCHED_SD_ROOT="$BUILD_DIR/patched-sd-src"
  echo "LLMEDGE_SDCPP_USE_MODS=1: creating patched stable-diffusion.cpp source tree at $PATCHED_SD_ROOT"
  rm -rf "$PATCHED_SD_ROOT"
  mkdir -p "$PATCHED_SD_ROOT"

  # Copy upstream sources into the patched tree.
  cp -a "$ROOT_DIR/stable-diffusion.cpp/." "$PATCHED_SD_ROOT/"

  # Overlay selected modified files (stored in mods/).
  #
  # IMPORTANT: `mods/` is not necessarily in sync with the current upstream submodule
  # and may contain files that won't compile against newer stable-diffusion.cpp.
  # Therefore, we only overlay explicitly requested files.
  #
  # Default overlays: only apply files that exist in mods/ by default.
  # If you add additional overlays (e.g. wan.hpp), set LLMEDGE_SDCPP_MODS_FILES accordingly.
  _DEFAULT_MODS_FILES="stable-diffusion.cpp,stable-diffusion.h,ggml_extend.hpp,util.cpp,ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp"
  if [[ -f "$ROOT_DIR/mods/wan.hpp" ]]; then
    _DEFAULT_MODS_FILES="wan.hpp,$_DEFAULT_MODS_FILES"
  fi
  MODS_FILES_RAW="${LLMEDGE_SDCPP_MODS_FILES:-$_DEFAULT_MODS_FILES}"
  # Map filenames to their new locations in the upstream source tree (src/ and include/)
  if [[ -d "$ROOT_DIR/mods" ]]; then
    IFS=',' read -r -a MODS_FILES <<< "$MODS_FILES_RAW"
    for f in "${MODS_FILES[@]}"; do
      f_trimmed="${f//[[:space:]]/}"
      [[ -z "$f_trimmed" ]] && continue
      
      # Determine target subdirectory based on file extension or specific mapping
      if [[ "$f_trimmed" == *.h ]]; then
        # Check if it should go to include/ or src/
        if [[ -f "$PATCHED_SD_ROOT/include/$f_trimmed" ]]; then
          target_path="include/$f_trimmed"
        else
          target_path="src/$f_trimmed"
        fi
      elif [[ "$f_trimmed" == *.cpp ]] || [[ "$f_trimmed" == *.hpp ]] || [[ "$f_trimmed" == *.inl ]]; then
        target_path="src/$f_trimmed"
      else
        target_path="$f_trimmed" # Fallback for nested paths
      fi
      
      # Special case for stable-diffusion.h which is in include/
      if [[ "$f_trimmed" == "stable-diffusion.h" ]]; then
        target_path="include/stable-diffusion.h"
      fi

      if [[ -f "$ROOT_DIR/mods/$f_trimmed" ]]; then
        echo "Overlaying mods/$f_trimmed -> patched tree at $target_path"
        mkdir -p "$(dirname "$PATCHED_SD_ROOT/$target_path")"
        cp -a "$ROOT_DIR/mods/$f_trimmed" "$PATCHED_SD_ROOT/$target_path"
      else
        echo "Warning: mods/$f_trimmed not found; skipping"
      fi
    done
  else
    echo "Warning: mods/ directory not found; continuing with pure upstream sources"
  fi

  CMAKE_EXTRA_ARGS+=("-DSD_ROOT_OVERRIDE=$PATCHED_SD_ROOT")
fi

# Try to find the CMake source root; fall back to scripts/jni-desktop if root has no CMakeLists
SRC_DIR="$ROOT_DIR"
if [[ ! -f "$SRC_DIR/CMakeLists.txt" ]]; then
  echo "No CMakeLists.txt detected at repo root; trying scripts/jni-desktop as CMake source"
  SRC_DIR="$ROOT_DIR/scripts/jni-desktop"
fi

# Configure CMake for a host build: enable Vulkan and WAN support.
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DSD_VULKAN=ON \
  -DWAN_SUPPORT=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPDLOG_FMT_EXTERNAL=ON \
  -DGGML_SKIP_OSX_FEATURES=ON \
  -DSDC_TEST_DESKTOP_JNI=ON \
  "${CMAKE_EXTRA_ARGS[@]}"

cmake --build "$BUILD_DIR" --target sdcpp --parallel $(nproc)

# Try to find libsdcpp
LIB_PATH=$(find "$BUILD_DIR" -type f -name 'libsdcpp*.so' -print -quit || true)
if [[ -z "$LIB_PATH" ]]; then
  echo "libsdcpp.so not found. CMake build might have failed or name different. Inspect $BUILD_DIR"
  exit 1
fi

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
  x86_64) ARCH_DIR="linux-x86_64" ;;
  aarch64|arm64) ARCH_DIR="linux-aarch64" ;;
  *) ARCH_DIR="linux-$ARCH" ;;
esac

mkdir -p "$ROOT_DIR/llmedge/build/native/$ARCH_DIR"
cp "$LIB_PATH" "$ROOT_DIR/llmedge/build/native/$ARCH_DIR/libsdcpp.so"

echo "Built and copied libsdcpp.so to llmedge/build/native/$ARCH_DIR/libsdcpp.so"
