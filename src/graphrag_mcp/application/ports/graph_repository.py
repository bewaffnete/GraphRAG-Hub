"""Graph repository port."""

from typing import Protocol

from graphrag_mcp.application.ports.parser_port import ParsedGraph
from graphrag_mcp.domain.value_objects.candidate_match import CandidateMatch


class GraphRepository(Protocol):
    def store_library_graph(self, parsed_graph: ParsedGraph) -> None:
        """Persist the parsed graph."""

    def search_candidates(
        self,
        *,
        query: str,
        graph_ids: list[str] | None,
        labels: list[str] | None,
        top_k: int,
        query_embedding: list[float] | None,
    ) -> list[CandidateMatch]:
        """Return compact candidate matches for the caller."""

    def list_graphs(self, *, name_prefix: str | None, status: str | None, limit: int) -> list[dict[str, object]]:
        """Return graph catalog items."""

    def get_nodes_by_ids(self, node_ids: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
        """Return nodes, edges between them, and missing ids."""

    def rebuild_embeddings(self, *, graph_id: str, embedding_provider) -> tuple[int, int]:
        """Recompute embeddings for a graph and persist them."""
