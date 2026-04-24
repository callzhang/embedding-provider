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
PORT="${PORT:-8000}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"

mkdir -p "$RUNTIME_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "instance already running: $INSTANCE_NAME (PID $(cat "$PID_FILE"))"
  exit 0
fi

find_listener_pid() {
  local pid=""
  if command -v lsof >/dev/null 2>&1; then
    pid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n1 || true)"
  fi
  if [[ -z "$pid" ]] && command -v ss >/dev/null 2>&1; then
    pid="$(ss -ltnp "( sport = :$PORT )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n1 || true)"
  fi
  printf '%s' "$pid"
}

LISTENER_PID="$(find_listener_pid)"
if [[ -n "$LISTENER_PID" ]]; then
  echo "port already in use for $INSTANCE_NAME: ${BIND_HOST}:${PORT} (PID $LISTENER_PID)" >&2
  if [[ -f "$PID_FILE" ]]; then
    echo "stale pid file: $PID_FILE -> $(cat "$PID_FILE")" >&2
  fi
  exit 1
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
  --host "$BIND_HOST" \
  --port "$PORT" \
  >"$LOG_FILE" 2>&1 &

echo $! >"$PID_FILE"
echo "started $INSTANCE_NAME with PID $(cat "$PID_FILE")"
echo "log: $LOG_FILE"

READY_TIMEOUT_SECONDS="${STARTUP_READY_TIMEOUT_SECONDS:-60}"
READY_URL="http://${BIND_HOST}:${PORT}/readyz"
START_DEADLINE=$((SECONDS + READY_TIMEOUT_SECONDS))

check_ready() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "$READY_URL" >/dev/null
    return $?
  fi
  python3 - "$READY_URL" <<'PY'
import sys
from urllib.request import urlopen

url = sys.argv[1]
with urlopen(url, timeout=2) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
}

until check_ready; do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "instance exited before becoming ready: $INSTANCE_NAME" >&2
    tail -n 50 "$LOG_FILE" || true
    exit 1
  fi
  if (( SECONDS >= START_DEADLINE )); then
    echo "instance did not become ready within ${READY_TIMEOUT_SECONDS}s: $INSTANCE_NAME" >&2
    tail -n 50 "$LOG_FILE" || true
    exit 1
  fi
  sleep 1
done

echo "ready: $READY_URL"
