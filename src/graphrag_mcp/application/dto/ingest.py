"""DTOs for library ingestion."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class IngestLibraryRequest:
    path: str
    library_name: str | None = None
    version: str | None = None
    ingest_mode: str = "parse_load_embed"
    embedding_mode: str = "enabled"


@dataclass(slots=True)
class EmbeddingSummary:
    enabled: bool
    embedded_nodes: int
    provider: str | None
    model: str | None


@dataclass(slots=True)
class IngestLibraryResponse:
    graph_id: str
    library_name: str
    version: str
    counts: dict[str, int]
    embedding_summary: EmbeddingSummary
    executed_stages: list[str]
    duration_ms: int
    warnings: list[str] = field(default_factory=list)
