"""Graph registry port."""

from typing import Protocol


class GraphRegistry(Protocol):
    def register_graph(self, graph_id: str, metadata: dict[str, object]) -> None:
        """Register graph metadata after ingest."""
