#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <env-file>" >&2
  exit 1
fi

ENV_FILE="$1"
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "env file not found: $ENV_FILE" >&2
  exit 1
fi

if [[ -z "$PROJECT_NAME" ]]; then
  PROJECT_NAME="$(grep '^COMPOSE_PROJECT_NAME=' "$ENV_FILE" | head -n1 | cut -d= -f2-)"
fi

if [[ -z "$PROJECT_NAME" ]]; then
  echo "COMPOSE_PROJECT_NAME must be set in $ENV_FILE" >&2
  exit 1
fi

docker compose --env-file "$ENV_FILE" -p "$PROJECT_NAME" up -d --build
