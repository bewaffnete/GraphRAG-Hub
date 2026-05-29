"""Parser port for source ingestion."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from graphrag_mcp.domain.entities.graph_edge import GraphEdge
from graphrag_mcp.domain.entities.graph_node import GraphNode


@dataclass(slots=True)
class ParsedGraph:
    library_name: str
    version: str
    root_path: Path
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ParserPort(Protocol):
    def parse_public_api(
        self,
        *,
        root_path: Path,
        library_name: str,
        version: str,
        graph_id: str,
    ) -> ParsedGraph:
        """Parse the public API and return graph-shaped entities."""
