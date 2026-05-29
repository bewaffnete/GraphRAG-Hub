"""Candidate match value object."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    node_id: str
    graph_id: str
    kind: str
    name: str
    qualified_name: str
    summary: str
    score: float
    source_path: str | None = None
    keyword_score: float = 0.0
    vector_score: float = 0.0
