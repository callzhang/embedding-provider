#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip

if ! "$VENV_DIR/bin/python" -c 'import torch' >/dev/null 2>&1; then
  "$VENV_DIR/bin/pip" install --index-url "$TORCH_INDEX_URL" torch
fi

"$VENV_DIR/bin/pip" install -e "$ROOT_DIR"
