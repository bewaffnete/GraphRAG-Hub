"""Retrieve details use case."""

from graphrag_mcp.application.dto.retrieve_details import (
    DetailEdge,
    DetailNode,
    RetrieveDetailsRequest,
    RetrieveDetailsResponse,
)
from graphrag_mcp.application.ports.graph_repository import GraphRepository
from graphrag_mcp.domain.services.context_compressor import ContextCompressor
from graphrag_mcp.domain.value_objects.detail_request import DetailRequest


class RetrieveDetailsUseCase:
    def __init__(self, *, repository: GraphRepository, context_compressor: ContextCompressor) -> None:
        self._repository = repository
        self._context_compressor = context_compressor

    def execute(self, request: RetrieveDetailsRequest) -> RetrieveDetailsResponse:
        detail_request = DetailRequest(
            node_ids=tuple(request.node_ids),
            include_relationships=request.include_relationships,
            include_docstring=request.include_docstring,
            include_logic_skeleton=request.include_logic_skeleton,
            include_source_excerpt=request.include_source_excerpt,
        )
        nodes, edges, missing = self._repository.get_nodes_by_ids(list(detail_request.node_ids))
        detail_nodes = [
            DetailNode(
                node_id=node["node_id"],
                graph_id=node["graph_id"],
                kind=node["kind"],
                name=node["name"],
                qualified_name=node["qualified_name"],
                signature=node.get("signature"),
                docstring=node.get("docstring") if detail_request.include_docstring else None,
                logic_skeleton=node.get("logic_skeleton") if detail_request.include_logic_skeleton else None,
                source_excerpt=node.get("source_excerpt") if detail_request.include_source_excerpt else None,
                metadata=node.get("metadata", {}),
            )
            for node in nodes
        ]
        detail_edges = [DetailEdge(**edge) for edge in edges] if detail_request.include_relationships else []
        expanded_context = self._context_compressor.compress(
            [f"{node.kind} {node.qualified_name}: {node.docstring or node.logic_skeleton or node.signature or ''}" for node in detail_nodes],
            max_items=len(detail_nodes),
        )
        return RetrieveDetailsResponse(
            nodes=detail_nodes,
            edges=detail_edges,
            expanded_context=expanded_context,
            missing_node_ids=missing,
        )
