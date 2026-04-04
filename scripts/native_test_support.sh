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

llmedge_copy_prebuilt_native_artifact() {
    local root_dir="$1"
    local target="$2"
    local native_dir="$3"
    local output_name
    local prebuilt_bin_dir
    local dep_name

    output_name="$(llmedge_native_output_name "$target")"
    prebuilt_bin_dir="$(llmedge_prebuilt_bin_dir "$root_dir")"
    [[ -d "$prebuilt_bin_dir" && -f "$prebuilt_bin_dir/$output_name" ]] || return 1

    echo "Copying prebuilt $output_name from $prebuilt_bin_dir"
    cp "$prebuilt_bin_dir/$output_name" "$native_dir/$output_name"
    while IFS= read -r dep_name; do
        [[ -n "$dep_name" && -f "$prebuilt_bin_dir/$dep_name" ]] || continue
        echo "Copying dependent $dep_name from $prebuilt_bin_dir to $native_dir"
        cp "$prebuilt_bin_dir/$dep_name" "$native_dir/$dep_name"
    done < <(llmedge_native_runtime_deps "$target")
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
        echo "Found native library at $native_dir/$output_name"
        return 0
    fi

    if llmedge_copy_prebuilt_native_artifact "$root_dir" "$target" "$native_dir"; then
        return 0
    fi

    echo "Prebuilt libs not found. Attempting to build with scripts/build_native_linux.sh $target"
    if [[ -f "$root_dir/scripts/build_native_linux.sh" ]]; then
        "$root_dir/scripts/build_native_linux.sh" "$target"
        return 0
    fi

    echo "No build script found; please build $(llmedge_native_output_name "$target") for host and place it in $native_dir" >&2
    return 1
}
