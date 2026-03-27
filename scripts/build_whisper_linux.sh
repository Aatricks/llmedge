#!/usr/bin/env bash
set -euo pipefail

# Build whisper.cpp JNI library for Linux (x86_64).
# Places the resulting libwhisper_jni.so into build/native/linux-x86_64

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
BUILD_DIR="$ROOT_DIR/scripts/jni-desktop/build-whisper"
mkdir -p "$BUILD_DIR"
SRC_DIR="$ROOT_DIR/scripts/jni-desktop"

echo "Configuring whisper desktop JNI build..."
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SDCPP=OFF \
    -DBUILD_SMOLLM=OFF \
    -DWHISPER_DESKTOP_JNI=ON

echo "Building whisper_jni target..."
cmake --build "$BUILD_DIR" --target whisper_jni --parallel $(nproc)

# Copy the library to the native directory
mkdir -p "$ROOT_DIR/llmedge/build/native/linux-x86_64"
cp "$BUILD_DIR/libwhisper_jni.so" "$ROOT_DIR/llmedge/build/native/linux-x86_64/"

# Also copy to a more accessible location
mkdir -p "$ROOT_DIR/scripts/jni-desktop/build/bin"
cp "$BUILD_DIR/libwhisper_jni.so" "$ROOT_DIR/scripts/jni-desktop/build/bin/"

echo "Built and copied libwhisper_jni.so to llmedge/build/native/linux-x86_64/libwhisper_jni.so"
