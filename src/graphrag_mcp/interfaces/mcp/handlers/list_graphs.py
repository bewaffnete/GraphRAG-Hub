"""MCP list graphs handler."""

from graphrag_mcp.application.dto.list_graphs import ListGraphsRequest
from graphrag_mcp.application.use_cases.list_graphs import ListGraphsUseCase
from graphrag_mcp.interfaces.mcp.handlers.error_mapping import map_error


def handle_graphrag_list_graphs(payload: dict[str, object], use_case: ListGraphsUseCase) -> dict[str, object]:
    try:
        request = ListGraphsRequest(
            name_prefix=str(payload["name_prefix"]) if payload.get("name_prefix") is not None else None,
            status=str(payload["status"]) if payload.get("status") is not None else None,
            limit=int(payload.get("limit", 50)),
        )
        response = use_case.execute(request)
    except Exception as exc:
        return map_error(exc)
    return {
        "graphs": [
            {
                "graph_id": item.graph_id,
                "name": item.name,
                "version": item.version,
                "status": item.status,
                # "node_counts": item.node_counts,
                # "embedding_status": item.embedding_status,
                "updated_at": item.updated_at,
            }
            for item in response.graphs
        ],
        "count": response.count,
        "warnings": response.warnings,
    }
