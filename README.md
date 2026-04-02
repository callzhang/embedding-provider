# Embedding Provider

Standalone OpenAI-compatible embedding service for Stardust shared use.

## Purpose

- Run independently from `memory-connector`
- Host embedding models on a shared GPU server
- Reuse one service codebase for different embedder instances
- Start multiple isolated provider instances from different env files

## Runtime Model

The service is a thin OpenAI-compatible wrapper over one loaded embedding model. It does not carry `memory-connector` backend logic. To run multiple embedders, start multiple compose projects with different env files, ports, cache directories, and optional GPU pinning.

Use a user-writable cache path such as `./runtime-cache/<instance-name>` for host deployments. Avoid reusing Docker-created root-owned cache folders under `./data/`.

## Instance Examples

Example Jina v5 instance:

```bash
cp deployments/gpu4/jina-v5-nano.env.example deployments/gpu4/jina-v5-nano.env
docker compose --env-file deployments/gpu4/jina-v5-nano.env -p embedding-provider-jina-v5-nano up -d --build
```

Example Qwen instance:

```bash
cp deployments/gpu4/qwen3-embedding-0.6b.env.example deployments/gpu4/qwen3-embedding-0.6b.env
docker compose --env-file deployments/gpu4/qwen3-embedding-0.6b.env -p embedding-provider-qwen3-embedding-0-6b up -d --build
```

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
set -a && source .env && set +a
uvicorn provider.app:app --host 127.0.0.1 --port 8000
```

## OpenAI-Compatible API

```bash
curl -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jinaai/jina-embeddings-v5-text-nano",
    "input": ["Alice leads Project Alpha"]
  }'
```

## Stardust GPU4

- Repo target: `stardust@stardust-gpu4:~/Projects/embedding-provider`
- Private bind: prefer `127.0.0.1` on the host, one port per instance
- Keep `memory-connector` pointing at the chosen provider instance through `GRAPHITI_EMBEDDER_BASE_URL`
- On the current `gpu4` host, the recommended path is the host-venv scripts under `scripts/`, because Docker GPU runtime is not configured.
- For public exposure, bind the model server to `127.0.0.1`, enforce `API_KEY`, and publish HTTPS through a Cloudflare named tunnel.

## Current Production Endpoint

- Public base URL: `https://embed.preseen.ai/v1`
- Current model: `jinaai/jina-embeddings-v5-text-nano`
- Current service bind on `gpu4`: `127.0.0.1:7997`
- Provider defaults:
  - `MAX_LENGTH=8192`
  - `MAX_BATCH_SIZE=8`
  - `DEFAULT_DIMENSIONS=768`

## Remote Sync

```bash
bash scripts/deploy_gpu4.sh
```

This syncs the repo to `stardust-gpu4` without forcing one fixed instance to start.

To start an instance on the server:

```bash
ssh stardust-gpu4-stardust
cd ~/Projects/embedding-provider
./scripts/start_host_instance.sh deployments/gpu4/jina-v5-nano.env
```

To expose that instance on the public internet with a fixed hostname on a NATed host like `gpu4`:

```bash
./scripts/start_public_cloudflared.sh deployments/gpu4/jina-v5-nano.env
```

Set these values in `deployments/gpu4/jina-v5-nano.env` before starting the tunnel:

```bash
BIND_HOST=127.0.0.1
API_KEY=change-me
PUBLIC_UPSTREAM=127.0.0.1:7997
```

Create the named tunnel and DNS route from a machine already logged into Cloudflare:

```bash
cloudflared tunnel create embedding-provider-preseen-ai
cloudflared tunnel route dns embedding-provider-preseen-ai embed.preseen.ai
cloudflared tunnel token embedding-provider-preseen-ai
```

Run the token on `gpu4`:

```bash
docker rm -f embedding-provider-cloudflared >/dev/null 2>&1 || true
docker run -d \
  --name embedding-provider-cloudflared \
  --restart unless-stopped \
  --network host \
  cloudflare/cloudflared:latest \
  tunnel --no-autoupdate run --token <TOKEN> --url http://127.0.0.1:7997
```

Operational checks:

```bash
curl https://embed.preseen.ai/healthz
curl -H "Authorization: Bearer <API_KEY>" https://embed.preseen.ai/v1/models
docker logs --tail 50 embedding-provider-cloudflared
```

If you have a directly routable public IP and want a host-local reverse proxy instead of Cloudflare Tunnel, the repo also includes `./scripts/start_public_caddy.sh` plus `deployments/gpu4/public.Caddyfile`.

The host scripts will auto-create `deployments/.../*.env` from the matching
`.env.example` on first run. To inspect or stop one instance:

```bash
./scripts/status_host_instance.sh deployments/gpu4/jina-v5-nano.env
./scripts/stop_host_instance.sh deployments/gpu4/jina-v5-nano.env
```
