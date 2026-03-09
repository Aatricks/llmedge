#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="$ROOT_DIR/llmedge-examples"
LIB_AAR="$ROOT_DIR/llmedge/build/outputs/aar/llmedge-release.aar"
TARGET_AAR="$EXAMPLES_DIR/app/libs/llmedge-release.aar"
EXAMPLE_TASKS=${LLMEDGE_EXAMPLES_GRADLE_TASKS:-":app:assembleDebug"}

if [[ "${LLMEDGE_SKIP_LIBRARY_BUILD:-false}" == "true" ]]; then
	if [[ ! -f "$LIB_AAR" ]]; then
		echo "LLMEDGE_SKIP_LIBRARY_BUILD=true was set but $LIB_AAR does not exist." >&2
		exit 1
	fi
	echo "Reusing existing llmedge release AAR..."
else
	echo "Building llmedge release AAR..."
	(cd "$ROOT_DIR" && ./gradlew --no-daemon :llmedge:assembleRelease)
fi

mkdir -p "$(dirname "$TARGET_AAR")"
cp "$LIB_AAR" "$TARGET_AAR"

read -r -a example_tasks <<< "$EXAMPLE_TASKS"

echo "Building llmedge-examples against the fresh AAR (${example_tasks[*]})..."
(cd "$EXAMPLES_DIR" && ./gradlew --no-daemon "${example_tasks[@]}")

echo "Example validation completed successfully."