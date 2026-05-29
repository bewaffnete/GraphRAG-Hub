#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${GRAPHRAG_EMBEDDING_MODEL:-qwen3-embedding:0.6b}"
LOCAL_OLLAMA_URL="${OLLAMA_HOST_URL:-http://localhost:11434}"

cd "${PROJECT_DIR}"

docker compose build mcp >&2
docker compose up -d neo4j >&2

if curl -fsS "${LOCAL_OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  export GRAPHRAG_EMBEDDING_BASE_URL="http://host.docker.internal:11434/api"
else
  docker compose up -d ollama >&2
  docker compose exec -T ollama ollama pull "${MODEL}" >&2
  export GRAPHRAG_EMBEDDING_BASE_URL="http://ollama:11434/api"
fi

docker compose up -d mcp >&2
exec docker exec -i graphrag-app graph-rag-mcp
