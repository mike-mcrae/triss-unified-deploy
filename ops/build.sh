#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRISS_DATA_DIR="${TRISS_DATA_DIR:-$ROOT_DIR/data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[build] ROOT_DIR=$ROOT_DIR"
echo "[build] TRISS_DATA_DIR=$TRISS_DATA_DIR"
echo "[build] Running pipeline sync..."

"$PYTHON_BIN" "$ROOT_DIR/pipeline/run_pipeline.py" "$@"

echo "[build] Final artifacts (depth<=3):"
find "$TRISS_DATA_DIR/final" -maxdepth 3 -type f | sort || true
