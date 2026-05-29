"""Retrieval query value object."""

from dataclasses import dataclass
from typing import ClassVar

from graphrag_mcp.domain.exceptions import GraphragError


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    query: str
    graph_id: str | None = None
    graph_ids: tuple[str, ...] = ()
    top_k: int = 8
    labels: tuple[str, ...] = ()
    traversal_depth: int = 2

    MIN_TOP_K: ClassVar[int] = 1
    MAX_TOP_K: ClassVar[int] = 50

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise GraphragError(code="INVALID_INPUT", message="Query must not be empty.", hint="Provide a concise technical query.")
        if self.graph_id and self.graph_ids:
            raise GraphragError(
                code="INVALID_INPUT",
                message="graph_id and graph_ids cannot be used together.",
                hint="Choose either one graph_id or a list of graph_ids.",
            )
        if not (self.MIN_TOP_K <= self.top_k <= self.MAX_TOP_K):
            raise GraphragError(
                code="INVALID_INPUT",
                message=f"top_k must be between {self.MIN_TOP_K} and {self.MAX_TOP_K}.",
                hint="Use a compact shortlist size for candidate retrieval.",
            )
