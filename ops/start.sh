#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRISS_DATA_DIR="${TRISS_DATA_DIR:-$ROOT_DIR/data}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[start] ROOT_DIR=$ROOT_DIR"
echo "[start] TRISS_DATA_DIR=$TRISS_DATA_DIR"
echo "[start] Final artifacts (depth<=3):"
find "$TRISS_DATA_DIR/final" -maxdepth 3 -type f | sort || true

echo "[start] Starting backend on ${HOST}:${PORT}"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m uvicorn app.backend.main:app --host "$HOST" --port "$PORT"
