#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CMAKE_TARGETS_FILE="${ROOT_DIR}/llmedge/src/main/cpp/cmake/llmedge-targets.cmake"
DEFAULT_OUT_FILE="${ROOT_DIR}/llmedge/src/main/java/io/aatricks/llmedge/core/NativeTargetNames.kt"
OUT_FILE="${1:-${DEFAULT_OUT_FILE}}"

if [[ ! -f "${CMAKE_TARGETS_FILE}" ]]; then
  echo "Missing target definition file: ${CMAKE_TARGETS_FILE}" >&2
  exit 1
fi

mapfile -t target_lines < <(sed -nE 's/^set\((LLMEDGE_TARGET_[A-Z0-9_]+) "([^"]+)"\)$/\1=\2/p' "${CMAKE_TARGETS_FILE}")

if [[ ${#target_lines[@]} -eq 0 ]]; then
  echo "No LLMEDGE_TARGET_* constants found in ${CMAKE_TARGETS_FILE}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_FILE}")"

{
  echo "package io.aatricks.llmedge.core"
  echo
  echo "/**"
  echo " * Generated from src/main/cpp/cmake/llmedge-targets.cmake."
  echo " * Do not edit manually; run scripts/generate_native_target_names.sh."
  echo " */"
  echo "internal object NativeTargetNames {"
  for line in "${target_lines[@]}"; do
    var_name="${line%%=*}"
    value="${line#*=}"
    kotlin_name="${var_name#LLMEDGE_TARGET_}"
    echo "    const val ${kotlin_name} = \"${value}\""
  done
  echo "}"
} > "${OUT_FILE}"

echo "Generated ${OUT_FILE}"
