"""MCP retrieve candidates handler."""

from graphrag_mcp.application.dto.retrieve_candidates import RetrieveCandidatesRequest
from graphrag_mcp.application.use_cases.retrieve_candidates import RetrieveCandidatesUseCase
from graphrag_mcp.interfaces.mcp.handlers.error_mapping import map_error


def handle_graphrag_retrieve_candidates(payload: dict[str, object], use_case: RetrieveCandidatesUseCase) -> dict[str, object]:
    try:
        request = RetrieveCandidatesRequest(
            query=str(payload["query"]),
            graph_id=str(payload["graph_id"]) if payload.get("graph_id") is not None else None,
            graph_ids=[str(item) for item in payload.get("graph_ids", [])] or None,
            top_k=int(payload.get("top_k", 8)),
            labels=[str(item) for item in payload.get("labels", [])] or None,
            traversal_depth=int(payload.get("traversal_depth", 2)),
        )
        response = use_case.execute(request)
    except Exception as exc:
        return map_error(exc)

    return {
        "query": response.query,
        "graph_scope": {
            "requested_graph_id": response.graph_scope.requested_graph_id,
            "requested_graph_ids": response.graph_scope.requested_graph_ids,
            "resolved_graph_ids": response.graph_scope.resolved_graph_ids,
        },
        "routed_graphs": [{"graph_id": item.graph_id, "score": item.score} for item in response.routed_graphs],
        "candidate_count": response.candidate_count,
        "top_matches": [
            {
                "node_id": item.node_id,
                "graph_id": item.graph_id,
                "kind": item.kind,
                "name": item.name,
                "qualified_name": item.qualified_name,
                "summary": item.summary,
                "score": item.score,
            }
            for item in response.top_matches
        ],
        "context_preview": response.context_preview,
        "warnings": response.warnings,
    }
