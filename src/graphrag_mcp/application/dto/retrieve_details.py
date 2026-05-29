"""DTOs for detail retrieval."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrieveDetailsRequest:
    node_ids: list[str]
    include_relationships: bool = True
    include_docstring: bool = True
    include_logic_skeleton: bool = True
    include_source_excerpt: bool = False


@dataclass(slots=True)
class DetailNode:
    node_id: str
    graph_id: str
    kind: str
    name: str
    qualified_name: str
    signature: str | None
    docstring: str | None
    logic_skeleton: str | None
    source_excerpt: str | None
    metadata: dict[str, object]


@dataclass(slots=True)
class DetailEdge:
    source_node_id: str
    target_node_id: str
    type: str


@dataclass(slots=True)
class RetrieveDetailsResponse:
    nodes: list[DetailNode]
    edges: list[DetailEdge]
    expanded_context: str
    missing_node_ids: list[str]
    warnings: list[str] = field(default_factory=list)
