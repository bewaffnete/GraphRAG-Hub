"""Embedding provider port."""

from typing import Protocol

from graphrag_mcp.application.ports.parser_port import ParsedGraph


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dimensions: int
    schema_version: str

    def embed_graph(self, parsed_graph: ParsedGraph) -> int:
        """Return the number of embedded nodes."""

    def embed_text(self, text: str) -> list[float]:
        """Return a deterministic embedding vector for the given text."""
