#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="$ROOT_DIR/llmedge-examples"
EXAMPLE_TASKS=${LLMEDGE_EXAMPLES_GRADLE_TASKS:-":app:assembleDebug"}

if [[ "${LLMEDGE_SKIP_LIBRARY_BUILD:-false}" == "true" ]]; then
	echo "Skipping :llmedge rebuild and reusing the existing local checkout..."
else
	echo "Building llmedge library..."
	(cd "$ROOT_DIR" && ./gradlew --no-daemon :llmedge:assembleRelease)
fi

read -r -a example_tasks <<< "$EXAMPLE_TASKS"

echo "Building llmedge-examples against the local :llmedge composite build (${example_tasks[*]})..."
(cd "$EXAMPLES_DIR" && ./gradlew --no-daemon "${example_tasks[@]}") || {
	echo "ERROR: llmedge-examples failed to compile against the local :llmedge checkout." >&2
	echo "This likely means a public API change in :llmedge broke the examples." >&2
	echo "Fix the examples or revert the API change before merging." >&2
	exit 1
}

# Run lint if available (non-fatal — reports warnings but doesn't block)
if [[ "${LLMEDGE_EXAMPLES_LINT:-true}" == "true" ]]; then
	echo "Running lint on llmedge-examples..."
	(cd "$EXAMPLES_DIR" && ./gradlew --no-daemon :app:lintDebug) || {
		echo "WARNING: Lint found issues in llmedge-examples (see report above)." >&2
		# Non-fatal: lint issues don't block the build
	}
fi

echo "Example validation completed successfully."
