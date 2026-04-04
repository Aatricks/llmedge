#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
"$ROOT_DIR/scripts/run_native_e2e.sh" text
