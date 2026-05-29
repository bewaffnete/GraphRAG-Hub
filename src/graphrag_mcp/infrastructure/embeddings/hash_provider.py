"""Deterministic hash embedding provider."""

import math

from graphrag_mcp.application.ports.parser_port import ParsedGraph


class HashEmbeddingProvider:
    provider_name = "hash"
    model_name = "deterministic-hash"
    dimensions = 32
    schema_version = "v1"

    def embed_graph(self, parsed_graph: ParsedGraph) -> int:
        embedded = 0
        for node in parsed_graph.nodes:
            if node.kind not in {"Module", "Class", "Function", "Example"}:
                continue
            embedding_text = _compose_embedding_text(node)
            vector = self.embed_text(embedding_text)
            node.metadata.update(
                {
                    "embedding": vector,
                    "embedding_provider": self.provider_name,
                    "embedding_model": self.model_name,
                    "embedding_dimensions": self.dimensions,
                    "embedding_schema_version": self.schema_version,
                    "embedding_text": embedding_text,
                }
            )
            embedded += 1
        return embedded

    def embed_text(self, text: str) -> list[float]:
        buckets = [0.0] * self.dimensions
        for token in _tokenize(text):
            bucket = hash(token) % self.dimensions
            buckets[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0:
            return buckets
        return [round(value / norm, 6) for value in buckets]


def _compose_embedding_text(node) -> str:
    parts = [
        node.kind,
        node.qualified_name,
        node.summary or "",
        node.signature or "",
        node.docstring or "",
    ]
    return " | ".join(part for part in parts if part).strip()


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().replace("|", " ").replace("(", " ").replace(")", " ").replace(",", " ").split() if token]
