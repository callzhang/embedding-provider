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

if [[ ! -f "$PID_FILE" ]]; then
  if [[ -n "$LISTENER_PID" ]]; then
    echo "status=running_without_pidfile instance=$INSTANCE_NAME pid=$LISTENER_PID bind=${BIND_HOST}:${PORT}"
    exit 0
  fi
  echo "status=stopped instance=$INSTANCE_NAME bind=${BIND_HOST}:${PORT}"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if ! kill -0 "$PID" 2>/dev/null; then
  if [[ -n "$LISTENER_PID" ]]; then
    echo "status=stale_pid_with_listener instance=$INSTANCE_NAME pid=$PID listener_pid=$LISTENER_PID bind=${BIND_HOST}:${PORT}"
    exit 1
  fi
  echo "status=stale_pid instance=$INSTANCE_NAME pid=$PID bind=${BIND_HOST}:${PORT}"
  exit 1
fi

echo "status=running instance=$INSTANCE_NAME pid=$PID bind=${BIND_HOST}:${PORT}"
if command -v ss >/dev/null 2>&1; then
  ss -ltnp | grep ":${PORT} " || true
fi
if command -v curl >/dev/null 2>&1; then
  echo "--- readyz ---"
  curl -fsS "http://${BIND_HOST}:${PORT}/readyz" || true
  echo
  echo "--- healthz ---"
  curl -fsS "http://${BIND_HOST}:${PORT}/healthz" || true
  echo
fi
if [[ -f "$LOG_FILE" ]]; then
  tail -n 20 "$LOG_FILE"
fi
