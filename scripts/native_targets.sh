#!/usr/bin/env bash

readonly -a LLMEDGE_DESKTOP_TARGETS=(smollm whisper sdcpp bark)
readonly -a LLMEDGE_CI_TARGETS=(whisper sdcpp smollm)

llmedge_is_known_native_target() {
    case "${1:-}" in
        smollm|whisper|sdcpp|bark) return 0 ;;
        *) return 1 ;;
    esac
}

llmedge_native_build_dir_name() {
    printf 'build-%s\n' "$1"
}

llmedge_native_cmake_target() {
    case "$1" in
        smollm) printf 'smollm\n' ;;
        whisper) printf 'whisper_jni\n' ;;
        sdcpp) printf 'sdcpp\n' ;;
        bark) printf 'bark_jni\n' ;;
        *)
            echo "Unknown native target: $1" >&2
            return 1
            ;;
    esac
}

llmedge_native_output_name() {
    case "$1" in
        smollm) printf 'libsmollm.so\n' ;;
        whisper) printf 'libwhisper_jni.so\n' ;;
        sdcpp) printf 'libsdcpp.so\n' ;;
        bark) printf 'libbark_jni.so\n' ;;
        *)
            echo "Unknown native target: $1" >&2
            return 1
            ;;
    esac
}

llmedge_native_alias_outputs() {
    case "$1" in
        smollm)
            printf 'libsmollm_v7a.so\n'
            printf 'libsmollm_v8.so\n'
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
