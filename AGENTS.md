# AGENTS.md

## Cursor Cloud specific instructions

This repo is a personal chat assistant: a **FastAPI backend** (pydantic-ai agent) that
runs the LLM inside a **Restate** durable-execution service and stores long-term memory
in **Neo4j** (via Graphiti). An optional **Next.js** frontend (`frontend/`) provides a
CopilotKit/AG-UI chat UI. See `CLAUDE.md` for the standard uv/ruff/pytest commands and
`chat_app.py`'s module docstring for the intended run order.

### Services and how to run them

The update script only refreshes dependencies (`uv sync`, frontend `npm install`). It does
NOT start Docker or any service — start those yourself each session:

| Service | Start command | Port |
|---|---|---|
| Docker daemon | `sudo dockerd` (run in a tmux/background session; not running by default) | — |
| Neo4j + Restate | `sudo docker compose up -d neo4j restate` | 7474/7687, 8080/9070 |
| FastAPI backend | `uv run uvicorn chat_app:app --host 0.0.0.0 --port 8000 --reload` | 8000 |
| Next.js frontend | `npm --prefix frontend run dev` | 3000 |

- Lint: `uv run ruff check .` and `npm --prefix frontend run lint`.
- Tests: `uv run pytest` (no API keys or database needed — uses pydantic-ai `TestModel`).

### Non-obvious gotchas

- **Docker 29 needs fuse-overlayfs + containerd-snapshotter disabled** in
  `/etc/docker/daemon.json` (already configured and persisted in the VM snapshot). If the
  daemon won't start with an overlay error, verify that file still has
  `"storage-driver": "fuse-overlayfs"` and `"features": {"containerd-snapshotter": false}`,
  and that iptables points at the legacy backend (`sudo update-alternatives --set iptables /usr/sbin/iptables-legacy`).
- **Restate registration is manual on Linux.** The app's lifespan auto-registration
  (`POST /deployments` in `chat_app.py`) does NOT work here because (a) it runs before
  uvicorn is listening, and (b) Restate defaults to HTTP/2 while uvicorn serves HTTP/1.1.
  After the backend is up, register manually with `use_http_11` and the compose network
  gateway:
  ```bash
  GW=$(sudo docker network inspect workspace_default -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}')
  curl -X POST http://localhost:9070/deployments -H 'Content-Type: application/json' \
       -d "{\"uri\": \"http://$GW:8000/restate-service\", \"use_http_11\": true}"
  ```
  Verify with `curl -s localhost:9070/services` (expect `ChatService` and `MemoryService`).
- **`SERVICE_ENDPOINT` must be the compose network gateway**, not `172.17.0.1`. `docker compose`
  creates the `workspace_default` bridge (gateway `172.18.0.1` in practice), not the default
  docker0 bridge. Restate calls the FastAPI deployment back over this gateway; set it in `.env`.
- **`.env` is gitignored** and must exist. `NEO4J_PASSWORD` must match between `.env` and the
  compose file. `TAVILY_API_KEY` must be set to *something* (even a dummy) or the backend
  fails at import time (`restate_service.py` reads it at module load).
- **Real chat needs real API keys**: `ANTHROPIC_API_KEY` (LLM), `TAVILY_API_KEY` (web search),
  `OPENAI_API_KEY` (Graphiti embeddings/reranking). With placeholder keys the whole pipeline
  runs but the model call returns `401 invalid x-api-key`; Restate then retries the invocation
  indefinitely — cancel a stuck one with
  `sudo docker exec workspace-restate-1 restate --yes invocations cancel --kill <inv_id>`.
- Neo4j `EquivalentSchemaRuleAlreadyExists` warnings on backend startup are benign (Graphiti's
  idempotent index creation), not failures.
