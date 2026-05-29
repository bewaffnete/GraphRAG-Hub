$ErrorActionPreference = "Stop"

$ProjectDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$Model = if ($env:GRAPHRAG_EMBEDDING_MODEL) { $env:GRAPHRAG_EMBEDDING_MODEL } else { "qwen3-embedding:0.6b" }
$LocalOllamaUrl = if ($env:OLLAMA_HOST_URL) { $env:OLLAMA_HOST_URL } else { "http://localhost:11434" }

Set-Location $ProjectDir

docker compose build mcp 1>&2
docker compose up -d neo4j 1>&2

try {
    Invoke-WebRequest -Uri "$LocalOllamaUrl/api/tags" -UseBasicParsing -TimeoutSec 2 | Out-Null
    $env:GRAPHRAG_EMBEDDING_BASE_URL = "http://host.docker.internal:11434/api"
}
catch {
    docker compose up -d ollama 1>&2
    docker compose exec -T ollama ollama pull $Model 1>&2
    $env:GRAPHRAG_EMBEDDING_BASE_URL = "http://ollama:11434/api"
}

docker compose up -d mcp 1>&2
docker exec -i graphrag-app graph-rag-mcp
