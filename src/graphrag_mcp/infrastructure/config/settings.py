"""Typed settings for bootstrap."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppSettings:
    app_name: str = "graphrag-mcp"
    version: str = "0.1.0"


@dataclass(frozen=True, slots=True)
class Neo4jSettings:
    backend: str = "memory"
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "neo4j"
    database: str = "neo4j"
    vector_index_name: str = "graph_node_embedding_index"
    fulltext_index_name: str = "graph_node_fulltext_index"


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    provider: str = "hash"
    model: str = "deterministic-hash"
    enabled: bool = False
    base_url: str = "http://localhost:11434/api"
    timeout_seconds: float = 30.0
    schema_version: str = "v1"
    api_key: str | None = None



@dataclass(frozen=True, slots=True)
class RegistrySettings:
    backend: str = "in_memory"


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str = "INFO"
    format: str = "human"
