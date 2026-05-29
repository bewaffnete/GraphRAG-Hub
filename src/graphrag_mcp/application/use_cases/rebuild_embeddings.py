"""Rebuild embeddings use case."""

import time

from graphrag_mcp.application.dto.rebuild_embeddings import RebuildEmbeddingsRequest, RebuildEmbeddingsResponse
from graphrag_mcp.application.ports.embedding_provider import EmbeddingProvider
from graphrag_mcp.application.ports.graph_repository import GraphRepository
from graphrag_mcp.domain.exceptions import EmbeddingProviderError


class RebuildEmbeddingsUseCase:
    def __init__(self, *, repository: GraphRepository, embedding_provider: EmbeddingProvider | None) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider

    def execute(self, request: RebuildEmbeddingsRequest) -> RebuildEmbeddingsResponse:
        if self._embedding_provider is None:
            raise EmbeddingProviderError(
                code="EMBEDDING_PROVIDER_ERROR",
                message="No embedding provider is configured.",
                hint="Enable embeddings and configure a provider before rebuilding.",
            )
        started = time.perf_counter()
        embedded_nodes, skipped_nodes = self._repository.rebuild_embeddings(
            graph_id=request.graph_id,
            embedding_provider=self._embedding_provider,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        return RebuildEmbeddingsResponse(
            graph_id=request.graph_id,
            embedded_nodes=embedded_nodes,
            skipped_nodes=skipped_nodes,
            provider=self._embedding_provider.provider_name,
            model=self._embedding_provider.model_name,
            schema_version=getattr(self._embedding_provider, "schema_version", None),
            duration_ms=duration_ms,
        )
