#!/usr/bin/env bash

LLMEDGE_NATIVE_TARGETS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLMEDGE_TARGETS_CMAKE_FILE="$LLMEDGE_NATIVE_TARGETS_SCRIPT_DIR/../llmedge/src/main/cpp/cmake/llmedge-targets.cmake"
LLMEDGE_NATIVE_TARGETS_GENERATOR="$LLMEDGE_NATIVE_TARGETS_SCRIPT_DIR/generate_native_target_names.sh"
LLMEDGE_GENERATED_TARGETS_FILE="$LLMEDGE_NATIVE_TARGETS_SCRIPT_DIR/../llmedge/build/generated/source/nativeTargetNames/main/shell/native-targets.sh"
LLMEDGE_GENERATED_KOTLIN_TARGETS_FILE="$LLMEDGE_NATIVE_TARGETS_SCRIPT_DIR/../llmedge/build/generated/source/nativeTargetNames/main/kotlin/io/aatricks/llmedge/core/NativeTargetNames.kt"

if [[ ! -f "$LLMEDGE_GENERATED_TARGETS_FILE" || "$LLMEDGE_TARGETS_CMAKE_FILE" -nt "$LLMEDGE_GENERATED_TARGETS_FILE" || "$LLMEDGE_NATIVE_TARGETS_GENERATOR" -nt "$LLMEDGE_GENERATED_TARGETS_FILE" ]]; then
    bash "$LLMEDGE_NATIVE_TARGETS_GENERATOR" "$LLMEDGE_GENERATED_KOTLIN_TARGETS_FILE" "$LLMEDGE_GENERATED_TARGETS_FILE" >/dev/null
fi

# shellcheck disable=SC1090
source "$LLMEDGE_GENERATED_TARGETS_FILE"

llmedge_is_known_native_target() {
    local target="${1:-}"
    local known
    for known in "${LLMEDGE_DESKTOP_TARGETS[@]}"; do
        [[ "$known" == "$target" ]] && return 0
    done
    return 1
}

llmedge_native_build_dir_name() {
    printf 'build-%s\n' "$1"
}

llmedge_native_cmake_target() {
    case "$1" in
        smollm) printf '%s\n' "$LLMEDGE_TARGET_SMOLLM" ;;
        whisper) printf '%s\n' "$LLMEDGE_TARGET_WHISPER_JNI" ;;
        sdcpp) printf '%s\n' "$LLMEDGE_TARGET_SDCPP" ;;
        bark) printf '%s\n' "$LLMEDGE_TARGET_BARK_JNI" ;;
        *)
            echo "Unknown native target: $1" >&2
            return 1
            ;;
    esac
}

llmedge_native_output_name() {
    printf 'lib%s.so\n' "$(llmedge_native_cmake_target "$1")"
}

llmedge_native_alias_outputs() {
    case "$1" in
        smollm)
            printf 'lib%s.so\n' "$LLMEDGE_TARGET_SMOLLM_V7A"
            printf 'lib%s.so\n' "$LLMEDGE_TARGET_SMOLLM_V8"
            ;;
    esac
}

llmedge_native_runtime_deps() {
    case "$1" in
        sdcpp) printf 'libstable-diffusion.so\n' ;;
    esac
}

llmedge_print_native_targets() {
    local target
    for target in "${LLMEDGE_DESKTOP_TARGETS[@]}"; do
        printf '  %s\n' "$target"
    done
    printf '  all\n'
}
