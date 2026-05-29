"""Retrieve candidates use case."""

from graphrag_mcp.application.dto.retrieve_candidates import (
    CandidateCard,
    GraphScope,
    RetrieveCandidatesRequest,
    RetrieveCandidatesResponse,
    RoutedGraph,
)
from graphrag_mcp.application.ports.embedding_provider import EmbeddingProvider
from graphrag_mcp.application.ports.graph_repository import GraphRepository
from graphrag_mcp.domain.services.context_compressor import ContextCompressor
from graphrag_mcp.domain.value_objects.retrieval_query import RetrievalQuery


class RetrieveCandidatesUseCase:
    def __init__(
        self,
        *,
        repository: GraphRepository,
        context_compressor: ContextCompressor,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._repository = repository
        self._context_compressor = context_compressor
        self._embedding_provider = embedding_provider

    def execute(self, request: RetrieveCandidatesRequest) -> RetrieveCandidatesResponse:
        retrieval_query = RetrievalQuery(
            query=request.query,
            graph_id=request.graph_id,
            graph_ids=tuple(request.graph_ids or ()),
            top_k=request.top_k,
            labels=tuple(request.labels or ()),
            traversal_depth=request.traversal_depth,
        )
        resolved_graph_ids = [retrieval_query.graph_id] if retrieval_query.graph_id else list(retrieval_query.graph_ids)
        query_embedding = self._embedding_provider.embed_text(retrieval_query.query) if self._embedding_provider is not None else None
        matches = self._repository.search_candidates(
            query=retrieval_query.query,
            graph_ids=resolved_graph_ids or None,
            labels=list(retrieval_query.labels) or None,
            top_k=retrieval_query.top_k,
            query_embedding=query_embedding,
        )
        top_matches = [
            CandidateCard(
                node_id=match.node_id,
                graph_id=match.graph_id,
                kind=match.kind,
                name=match.name,
                qualified_name=match.qualified_name,
                summary=match.summary,
                score=match.score,
            )
            for match in matches
        ]
        preview = self._context_compressor.compress(
            [f"{match.kind} {match.qualified_name}: {match.summary}" for match in top_matches],
            max_items=min(len(top_matches), 3),
        )
        routed_graph_scores: dict[str, float] = {}
        for match in matches:
            routed_graph_scores[match.graph_id] = max(routed_graph_scores.get(match.graph_id, 0.0), match.score)
        routed_graphs = [RoutedGraph(graph_id=graph_id, score=score) for graph_id, score in sorted(routed_graph_scores.items(), key=lambda item: item[1], reverse=True)]
        warnings: list[str] = []
        if self._embedding_provider is None:
            warnings.append("Vector search is disabled because no embedding provider is configured.")
        return RetrieveCandidatesResponse(
            query=retrieval_query.query,
            graph_scope=GraphScope(
                requested_graph_id=retrieval_query.graph_id,
                requested_graph_ids=list(retrieval_query.graph_ids),
                resolved_graph_ids=resolved_graph_ids,
            ),
            routed_graphs=routed_graphs,
            candidate_count=len(top_matches),
            top_matches=top_matches,
            context_preview=preview,
            warnings=warnings,
        )
