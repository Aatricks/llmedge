#!/usr/bin/env bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/native_targets.sh"

llmedge_native_arch_dir() {
    local arch
    arch=$(uname -m)
    case "$arch" in
        x86_64) printf 'linux-x86_64\n' ;;
        aarch64|arm64) printf 'linux-aarch64\n' ;;
        *) printf 'linux-%s\n' "$arch" ;;
    esac
}

llmedge_host_native_dir() {
    local root_dir="$1"
    printf '%s/llmedge/build/native/%s\n' "$root_dir" "$(llmedge_native_arch_dir)"
}

llmedge_prebuilt_bin_dir() {
    local root_dir="$1"
    printf '%s/scripts/jni-desktop/build/bin\n' "$root_dir"
}

llmedge_prebuilt_native_artifact_path() {
    local root_dir="$1"
    local target="$2"
    printf '%s/%s\n' "$(llmedge_prebuilt_bin_dir "$root_dir")" "$(llmedge_native_output_name "$target")"
}

llmedge_copy_prebuilt_native_artifact() {
    local root_dir="$1"
    local target="$2"
    local native_dir="$3"
    local output_name
    local prebuilt_bin_dir
    local prebuilt_artifact
    local dep_name

    output_name="$(llmedge_native_output_name "$target")"
    prebuilt_bin_dir="$(llmedge_prebuilt_bin_dir "$root_dir")"
    prebuilt_artifact="$(llmedge_prebuilt_native_artifact_path "$root_dir" "$target")"
    [[ -d "$prebuilt_bin_dir" && -f "$prebuilt_artifact" ]] || return 1
    if llmedge_host_native_artifact_is_stale "$root_dir" "$target" "$prebuilt_artifact"; then
        return 1
    fi

    echo "Copying prebuilt $output_name from $prebuilt_bin_dir"
    cp "$prebuilt_artifact" "$native_dir/$output_name"
    while IFS= read -r dep_name; do
        [[ -n "$dep_name" && -f "$prebuilt_bin_dir/$dep_name" ]] || continue
        echo "Copying dependent $dep_name from $prebuilt_bin_dir to $native_dir"
        cp "$prebuilt_bin_dir/$dep_name" "$native_dir/$dep_name"
    done < <(llmedge_native_runtime_deps "$target")
}

llmedge_native_source_paths() {
    local root_dir="$1"
    local target="$2"
    case "$target" in
        smollm)
            printf '%s\n' \
                "$root_dir/llmedge/src/main/cpp/LLMInference.cpp" \
                "$root_dir/llmedge/src/main/cpp/LLMInference.h" \
                "$root_dir/llmedge/src/main/cpp/llm_backend_support.cpp" \
                "$root_dir/llmedge/src/main/cpp/llm_backend_support.h" \
                "$root_dir/llmedge/src/main/cpp/smollm.cpp" \
                "$root_dir/llmedge/src/main/cpp/smollm_jni_completion.cpp" \
                "$root_dir/llmedge/src/main/cpp/smollm_jni_embeddings.cpp" \
                "$root_dir/llmedge/src/main/cpp/smollm_jni_load.cpp" \
                "$root_dir/llmedge/src/main/cpp/smollm_jni_state.cpp" \
                "$root_dir/llmedge/src/main/cpp/cmake/llmedge-common.cmake" \
                "$root_dir/llmedge/src/main/cpp/cmake/llmedge-smollm.cmake"
            ;;
        whisper)
            printf '%s\n' \
                "$root_dir/llmedge/src/main/cpp/whisper_jni.cpp" \
                "$root_dir/llmedge/src/main/cpp/whisper_jni_common.cpp" \
                "$root_dir/llmedge/src/main/cpp/whisper_jni_common.h" \
                "$root_dir/llmedge/src/main/cpp/cmake/llmedge-whisper.cmake" \
                "$root_dir/llmedge/src/main/java/io/aatricks/llmedge/speech/stt/Whisper.kt" \
                "$root_dir/llmedge/src/main/java/io/aatricks/llmedge/speech/stt/WhisperRuntimeSupport.kt"
            ;;
        sdcpp)
            printf '%s\n' \
                "$root_dir/llmedge/src/main/cpp/sdcpp_jni.cpp" \
                "$root_dir/llmedge/src/main/cpp/sdcpp_jni_common.cpp" \
                "$root_dir/llmedge/src/main/cpp/sdcpp_jni_condition.cpp" \
                "$root_dir/llmedge/src/main/cpp/sdcpp_jni_image.cpp" \
                "$root_dir/llmedge/src/main/cpp/sdcpp_jni_load.cpp" \
                "$root_dir/llmedge/src/main/cpp/sdcpp_jni_video.cpp" \
                "$root_dir/llmedge/src/main/cpp/cmake/llmedge-stable-diffusion.cmake"
            ;;
        bark)
            printf '%s\n' \
                "$root_dir/llmedge/src/main/cpp/bark_jni.cpp" \
                "$root_dir/llmedge/src/main/cpp/cmake/llmedge-bark.cmake"
            ;;
    esac
}

llmedge_host_native_artifact_is_stale() {
    local root_dir="$1"
    local target="$2"
    local artifact_path="$3"
    local source_path

    [[ -f "$artifact_path" ]] || return 0

    while IFS= read -r source_path; do
        [[ -n "$source_path" && -e "$source_path" ]] || continue
        if [[ "$source_path" -nt "$artifact_path" ]]; then
            echo "Native artifact is stale relative to $source_path"
            return 0
        fi
    done < <(llmedge_native_source_paths "$root_dir" "$target")

    return 1
}

llmedge_ensure_host_native_artifact() {
    local root_dir="$1"
    local target="$2"
    local native_dir="$3"
    local output_name

    llmedge_is_known_native_target "$target" || {
        echo "Unknown native target: $target" >&2
        return 1
    }

    output_name="$(llmedge_native_output_name "$target")"
    mkdir -p "$native_dir"

    if [[ -f "$native_dir/$output_name" ]]; then
        if ! llmedge_host_native_artifact_is_stale "$root_dir" "$target" "$native_dir/$output_name"; then
            echo "Found native library at $native_dir/$output_name"
            return 0
        fi
        rm -f "$native_dir/$output_name"
    fi

    if llmedge_copy_prebuilt_native_artifact "$root_dir" "$target" "$native_dir"; then
        return 0
    fi

    echo "Prebuilt libs not found or stale. Attempting to build with scripts/build_native_linux.sh $target"
    if [[ -f "$root_dir/scripts/build_native_linux.sh" ]]; then
        "$root_dir/scripts/build_native_linux.sh" "$target"
        return 0
    fi

    echo "No build script found; please build $(llmedge_native_output_name "$target") for host and place it in $native_dir" >&2
    return 1
}
