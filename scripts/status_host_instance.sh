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
PID_FILE="$ROOT_DIR/runtime/${INSTANCE_NAME}.pid"
LOG_FILE="$ROOT_DIR/runtime/${INSTANCE_NAME}.log"
PORT="${PORT:-8000}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "status=stopped instance=$INSTANCE_NAME"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "status=stale_pid instance=$INSTANCE_NAME pid=$PID"
  exit 1
fi

echo "status=running instance=$INSTANCE_NAME pid=$PID bind=${BIND_HOST}:${PORT}"
if command -v ss >/dev/null 2>&1; then
  ss -ltnp | grep ":${PORT} " || true
fi
if [[ -f "$LOG_FILE" ]]; then
  tail -n 20 "$LOG_FILE"
fi
