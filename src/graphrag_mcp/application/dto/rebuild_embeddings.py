"""DTOs for embedding rebuild operations."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class RebuildEmbeddingsRequest:
    graph_id: str
    provider: str | None = None
    model: str | None = None
    schema_version: str | None = None


@dataclass(slots=True)
class RebuildEmbeddingsResponse:
    graph_id: str
    embedded_nodes: int
    skipped_nodes: int
    provider: str | None
    model: str | None
    schema_version: str | None
    duration_ms: int
    warnings: list[str] = field(default_factory=list)
