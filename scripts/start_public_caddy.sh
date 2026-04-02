#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <env-file>" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$1"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "env file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

PUBLIC_HOSTNAME="${PUBLIC_HOSTNAME:-}"
PUBLIC_BIND="${PUBLIC_BIND:-}"
PUBLIC_UPSTREAM="${PUBLIC_UPSTREAM:-host.docker.internal:${PORT:-8000}}"
CONTAINER_NAME="${PUBLIC_PROXY_CONTAINER_NAME:-embedding-provider-caddy}"
CADDYFILE_PATH="${PUBLIC_CADDYFILE:-$ROOT_DIR/deployments/gpu4/public.Caddyfile}"
DATA_DIR="$ROOT_DIR/runtime/caddy-data"
CONFIG_DIR="$ROOT_DIR/runtime/caddy-config"

if [[ -z "$PUBLIC_HOSTNAME" ]]; then
  echo "PUBLIC_HOSTNAME must be set in $ENV_FILE" >&2
  exit 1
fi

if [[ -z "$PUBLIC_BIND" ]]; then
  echo "PUBLIC_BIND must be set in $ENV_FILE" >&2
  exit 1
fi

if [[ ! -f "$CADDYFILE_PATH" ]]; then
  echo "Caddyfile not found: $CADDYFILE_PATH" >&2
  exit 1
fi

mkdir -p "$DATA_DIR" "$CONFIG_DIR"

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --add-host host.docker.internal:host-gateway \
  -e PUBLIC_HOSTNAME="$PUBLIC_HOSTNAME" \
  -e PUBLIC_BIND="$PUBLIC_BIND" \
  -e PUBLIC_UPSTREAM="$PUBLIC_UPSTREAM" \
  -p "${PUBLIC_BIND}:80:80" \
  -p "${PUBLIC_BIND}:443:443" \
  -v "$CADDYFILE_PATH:/etc/caddy/Caddyfile:ro" \
  -v "$DATA_DIR:/data" \
  -v "$CONFIG_DIR:/config" \
  caddy:2 >/dev/null

echo "started public proxy container=$CONTAINER_NAME hostname=$PUBLIC_HOSTNAME upstream=$PUBLIC_UPSTREAM"
