#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CMAKE_TARGETS_FILE="${ROOT_DIR}/llmedge/src/main/cpp/cmake/llmedge-targets.cmake"
DEFAULT_OUT_FILE="${ROOT_DIR}/llmedge/build/generated/source/nativeTargetNames/main/kotlin/io/aatricks/llmedge/core/NativeTargetNames.kt"
DEFAULT_SHELL_OUT_FILE="${ROOT_DIR}/llmedge/build/generated/source/nativeTargetNames/main/shell/native-targets.sh"
OUT_FILE="${1:-${DEFAULT_OUT_FILE}}"
SHELL_OUT_FILE="${2:-${DEFAULT_SHELL_OUT_FILE}}"

if [[ ! -f "${CMAKE_TARGETS_FILE}" ]]; then
  echo "Missing target definition file: ${CMAKE_TARGETS_FILE}" >&2
  exit 1
fi

mapfile -t target_lines < <(sed -nE 's/^set\((LLMEDGE_TARGET_[A-Z0-9_]+) "([^"]+)"\)$/\1=\2/p' "${CMAKE_TARGETS_FILE}")
mapfile -t list_lines < <(sed -nE 's/^set\((LLMEDGE_(DESKTOP|CI)_TARGETS) "([^"]*)"\)$/\1=\3/p' "${CMAKE_TARGETS_FILE}")

if [[ ${#target_lines[@]} -eq 0 ]]; then
  echo "No LLMEDGE_TARGET_* constants found in ${CMAKE_TARGETS_FILE}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_FILE}")"
mkdir -p "$(dirname "${SHELL_OUT_FILE}")"

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

{
  echo "#!/usr/bin/env bash"
  echo "# Generated from llmedge/src/main/cpp/cmake/llmedge-targets.cmake."
  echo "# Do not edit manually; run scripts/generate_native_target_names.sh."
  for line in "${target_lines[@]}"; do
    var_name="${line%%=*}"
    value="${line#*=}"
    printf "readonly %s='%s'\n" "${var_name}" "${value}"
  done
  for line in "${list_lines[@]}"; do
    var_name="${line%%=*}"
    raw_values="${line#*=}"
    printf "readonly -a %s=(" "${var_name}"
    IFS=';' read -r -a items <<< "${raw_values}"
    for item in "${items[@]}"; do
      [[ -z "${item}" ]] && continue
      printf "'%s' " "${item}"
    done
    echo ")"
  done
} > "${SHELL_OUT_FILE}"

chmod +x "${SHELL_OUT_FILE}"

echo "Generated ${OUT_FILE}"
echo "Generated ${SHELL_OUT_FILE}"
