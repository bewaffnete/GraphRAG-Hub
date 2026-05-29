"""DTOs for graph catalog listing."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class ListGraphsRequest:
    name_prefix: str | None = None
    status: str | None = None
    limit: int = 50


@dataclass(slots=True)
class GraphCatalogItem:
    graph_id: str
    name: str
    version: str
    status: str
    node_counts: dict[str, int]
    embedding_status: dict[str, object]
    updated_at: str | None


@dataclass(slots=True)
class ListGraphsResponse:
    graphs: list[GraphCatalogItem]
    count: int
    warnings: list[str] = field(default_factory=list)
