"""Configuration loader."""

from dataclasses import dataclass
import os
from pathlib import Path

from graphrag_mcp.infrastructure.config.settings import (
    AppSettings,
    EmbeddingSettings,
    LoggingSettings,
    Neo4jSettings,
    RegistrySettings,
)


@dataclass(frozen=True, slots=True)
class SettingsBundle:
    app: AppSettings
    neo4j: Neo4jSettings
    embedding: EmbeddingSettings
    registry: RegistrySettings
    logging: LoggingSettings


def _load_dotenv_file(path: Path | None = None) -> None:
    dotenv_path = path or Path.cwd() / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_settings_from_env() -> SettingsBundle:
    _load_dotenv_file()
    return SettingsBundle(
        app=AppSettings(
            app_name=os.getenv("GRAPHRAG_APP_NAME", "graphrag-mcp"),
            version=os.getenv("GRAPHRAG_APP_VERSION", "0.1.0"),
        ),
        neo4j=Neo4jSettings(
            backend=os.getenv("GRAPHRAG_NEO4J_BACKEND", "memory"),
            uri=os.getenv("GRAPHRAG_NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("GRAPHRAG_NEO4J_USERNAME", "neo4j"),
            password=os.getenv("GRAPHRAG_NEO4J_PASSWORD", "neo4j"),
            database=os.getenv("GRAPHRAG_NEO4J_DATABASE", "neo4j"),
            vector_index_name=os.getenv("GRAPHRAG_NEO4J_VECTOR_INDEX_NAME", "graph_node_embedding_index"),
            fulltext_index_name=os.getenv("GRAPHRAG_NEO4J_FULLTEXT_INDEX_NAME", "graph_node_fulltext_index"),
        ),
        embedding=EmbeddingSettings(
            provider=os.getenv("GRAPHRAG_EMBEDDING_PROVIDER", "hash"),
            model=os.getenv("GRAPHRAG_EMBEDDING_MODEL", "deterministic-hash"),
            enabled=os.getenv("GRAPHRAG_EMBEDDINGS_ENABLED", "false").lower() == "true",
            base_url=os.getenv("GRAPHRAG_EMBEDDING_BASE_URL", "http://localhost:11434/api"),
            timeout_seconds=float(os.getenv("GRAPHRAG_EMBEDDING_TIMEOUT_SECONDS", "30")),
            schema_version=os.getenv("GRAPHRAG_EMBEDDING_SCHEMA_VERSION", "v1"),
            api_key=os.getenv("GRAPHRAG_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"),
        ),
        registry=RegistrySettings(
            backend=os.getenv("GRAPHRAG_REGISTRY_BACKEND", "in_memory"),
        ),
        logging=LoggingSettings(
            level=os.getenv("GRAPHRAG_LOG_LEVEL", "INFO"),
            format=os.getenv("GRAPHRAG_LOG_FORMAT", "human"),
        ),
    )
