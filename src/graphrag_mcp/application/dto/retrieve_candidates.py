"""DTOs for candidate retrieval."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrieveCandidatesRequest:
    query: str
    graph_id: str | None = None
    graph_ids: list[str] | None = None
    top_k: int = 8
    labels: list[str] | None = None
    traversal_depth: int = 2


@dataclass(slots=True)
class RoutedGraph:
    graph_id: str
    score: float


@dataclass(slots=True)
class CandidateCard:
    node_id: str
    graph_id: str
    kind: str
    name: str
    qualified_name: str
    summary: str
    score: float


@dataclass(slots=True)
class GraphScope:
    requested_graph_id: str | None
    requested_graph_ids: list[str]
    resolved_graph_ids: list[str]


@dataclass(slots=True)
class RetrieveCandidatesResponse:
    query: str
    graph_scope: GraphScope
    routed_graphs: list[RoutedGraph]
    candidate_count: int
    top_matches: list[CandidateCard]
    context_preview: str
    warnings: list[str] = field(default_factory=list)
