#!/usr/bin/env bash
set -euo pipefail

MODEL="${GRAPHRAG_EMBEDDING_MODEL:-qwen3-embedding:0.6b}"
LOCAL_OLLAMA_URL="${OLLAMA_HOST_URL:-http://localhost:11434}"

docker compose up -d --build neo4j

if curl -fsS "${LOCAL_OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "Local Ollama found at ${LOCAL_OLLAMA_URL}."
  echo "MCP containers will use GRAPHRAG_EMBEDDING_BASE_URL=http://host.docker.internal:11434/api."
  export GRAPHRAG_EMBEDDING_BASE_URL="http://host.docker.internal:11434/api"
else
  echo "Local Ollama was not found at ${LOCAL_OLLAMA_URL}; starting Ollama container."
  docker compose up -d ollama
  docker compose exec ollama ollama pull "${MODEL}"
  export GRAPHRAG_EMBEDDING_BASE_URL="http://ollama:11434/api"
fi

docker compose up -d --build
