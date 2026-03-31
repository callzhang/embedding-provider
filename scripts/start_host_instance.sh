#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <env-file>" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$1"

if [[ ! -f "$ENV_FILE" ]]; then
  if [[ -f "${ENV_FILE}.example" ]]; then
    cp "${ENV_FILE}.example" "$ENV_FILE"
    echo "created env file from example: $ENV_FILE"
  else
    echo "env file not found: $ENV_FILE" >&2
    exit 1
  fi
fi

set -a
source "$ENV_FILE"
set +a

INSTANCE_NAME="${COMPOSE_PROJECT_NAME:-$(basename "$ENV_FILE" .env)}"
RUNTIME_DIR="$ROOT_DIR/runtime"
PID_FILE="$RUNTIME_DIR/${INSTANCE_NAME}.pid"
LOG_FILE="$RUNTIME_DIR/${INSTANCE_NAME}.log"

mkdir -p "$RUNTIME_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "instance already running: $INSTANCE_NAME (PID $(cat "$PID_FILE"))"
  exit 0
fi

"$ROOT_DIR/scripts/bootstrap_venv.sh"

mkdir -p "$RUNTIME_DIR"

if [[ -n "${HF_CACHE_DIR:-}" ]]; then
  mkdir -p "$ROOT_DIR/${HF_CACHE_DIR#./}"
  export HF_HOME="$ROOT_DIR/${HF_CACHE_DIR#./}"
  export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

nohup "$ROOT_DIR/.venv/bin/uvicorn" provider.app:app \
  --app-dir "$ROOT_DIR" \
  --host "${BIND_HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  >"$LOG_FILE" 2>&1 &

echo $! >"$PID_FILE"
echo "started $INSTANCE_NAME with PID $(cat "$PID_FILE")"
echo "log: $LOG_FILE"
