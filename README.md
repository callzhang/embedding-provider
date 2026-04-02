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
- Default bind: Tailscale IP on the host, one port per instance
- Keep `memory-connector` pointing at the chosen provider instance through `GRAPHITI_EMBEDDER_BASE_URL`
- On the current `gpu4` host, the recommended path is the host-venv scripts under `scripts/`, because Docker GPU runtime is not configured.
- For public exposure, bind the model server to `127.0.0.1`, enforce `API_KEY`, and publish HTTPS through either a direct reverse proxy or the bundled Cloudflare quick tunnel helper.

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

To expose that instance on the public internet with HTTPS on a NATed host like `gpu4`:

```bash
./scripts/start_public_cloudflared.sh deployments/gpu4/jina-v5-nano.env
```

Set these values in `deployments/gpu4/jina-v5-nano.env` before starting the public tunnel:

```bash
BIND_HOST=127.0.0.1
API_KEY=change-me
PUBLIC_UPSTREAM=127.0.0.1:7997
```

If you have a directly routable public IP and want a fixed hostname with automatic TLS, the repo also includes `./scripts/start_public_caddy.sh` plus `deployments/gpu4/public.Caddyfile`.

The host scripts will auto-create `deployments/.../*.env` from the matching
`.env.example` on first run. To inspect or stop one instance:

```bash
./scripts/status_host_instance.sh deployments/gpu4/jina-v5-nano.env
./scripts/stop_host_instance.sh deployments/gpu4/jina-v5-nano.env
```
