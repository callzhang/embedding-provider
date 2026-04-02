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

CONTAINER_NAME="${PUBLIC_TUNNEL_CONTAINER_NAME:-embedding-provider-cloudflared}"
PUBLIC_UPSTREAM="${PUBLIC_UPSTREAM:-127.0.0.1:${PORT:-8000}}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --network host \
  cloudflare/cloudflared:latest tunnel --no-autoupdate --url "http://${PUBLIC_UPSTREAM}" >/dev/null

for _ in $(seq 1 30); do
  url="$(docker logs "$CONTAINER_NAME" 2>&1 | grep -Eo 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' | tail -n 1 || true)"
  if [[ -n "$url" ]]; then
    echo "started public tunnel container=$CONTAINER_NAME url=$url upstream=$PUBLIC_UPSTREAM"
    exit 0
  fi
  sleep 1
done

echo "cloudflared started but public URL was not found in logs" >&2
docker logs --tail 100 "$CONTAINER_NAME" >&2 || true
exit 1
