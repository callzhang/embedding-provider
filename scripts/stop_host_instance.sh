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

if [[ ! -f "$PID_FILE" ]]; then
  echo "no pid file for $INSTANCE_NAME"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
fi
rm -f "$PID_FILE"
echo "stopped $INSTANCE_NAME"
