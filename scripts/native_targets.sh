#!/usr/bin/env bash

LLMEDGE_NATIVE_TARGETS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLMEDGE_TARGETS_CMAKE_FILE="$LLMEDGE_NATIVE_TARGETS_SCRIPT_DIR/../llmedge/src/main/cpp/cmake/llmedge-targets.cmake"

llmedge_read_cmake_target_var() {
    local var_name="$1"
    sed -nE "s/^set\\(${var_name} \"([^\"]+)\"\\)$/\\1/p" "$LLMEDGE_TARGETS_CMAKE_FILE" | head -n 1
}

llmedge_read_cmake_list_var() {
    local var_name="$1"
    local raw
    raw="$(llmedge_read_cmake_target_var "$var_name")"
    if [[ -z "$raw" ]]; then
        return 0
    fi
    tr ';' '\n' <<<"$raw"
}

readonly LLMEDGE_TARGET_SMOLLM="$(llmedge_read_cmake_target_var LLMEDGE_TARGET_SMOLLM)"
readonly LLMEDGE_TARGET_SMOLLM_V7A="$(llmedge_read_cmake_target_var LLMEDGE_TARGET_SMOLLM_V7A)"
readonly LLMEDGE_TARGET_SMOLLM_V8="$(llmedge_read_cmake_target_var LLMEDGE_TARGET_SMOLLM_V8)"
readonly LLMEDGE_TARGET_SDCPP="$(llmedge_read_cmake_target_var LLMEDGE_TARGET_SDCPP)"
readonly LLMEDGE_TARGET_WHISPER_JNI="$(llmedge_read_cmake_target_var LLMEDGE_TARGET_WHISPER_JNI)"
readonly LLMEDGE_TARGET_BARK_JNI="$(llmedge_read_cmake_target_var LLMEDGE_TARGET_BARK_JNI)"

mapfile -t LLMEDGE_DESKTOP_TARGETS < <(llmedge_read_cmake_list_var LLMEDGE_DESKTOP_TARGETS)
mapfile -t LLMEDGE_CI_TARGETS < <(llmedge_read_cmake_list_var LLMEDGE_CI_TARGETS)

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
