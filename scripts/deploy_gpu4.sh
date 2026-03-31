#!/usr/bin/env bash
set -euo pipefail

REMOTE="stardust-gpu4-stardust"
REMOTE_DIR="~/Projects/embedding-provider"

ssh "$REMOTE" "mkdir -p $REMOTE_DIR"
rsync -az \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'data' \
  --exclude 'runtime' \
  --exclude 'runtime-cache' \
  --exclude 'deployments/**/*.env' \
  ./ "$REMOTE:$REMOTE_DIR/"

echo "Deployed to $REMOTE:$REMOTE_DIR"
