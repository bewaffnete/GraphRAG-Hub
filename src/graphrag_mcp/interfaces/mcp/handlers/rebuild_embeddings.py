"""MCP rebuild embeddings handler."""

from graphrag_mcp.application.dto.rebuild_embeddings import RebuildEmbeddingsRequest
from graphrag_mcp.application.use_cases.rebuild_embeddings import RebuildEmbeddingsUseCase
from graphrag_mcp.interfaces.mcp.handlers.error_mapping import map_error


def handle_graphrag_rebuild_embeddings(payload: dict[str, object], use_case: RebuildEmbeddingsUseCase) -> dict[str, object]:
    try:
        request = RebuildEmbeddingsRequest(
            graph_id=str(payload["graph_id"]),
            provider=str(payload["provider"]) if payload.get("provider") is not None else None,
            model=str(payload["model"]) if payload.get("model") is not None else None,
            schema_version=str(payload["schema_version"]) if payload.get("schema_version") is not None else None,
        )
        response = use_case.execute(request)
    except Exception as exc:
        return map_error(exc)
    return {
        "graph_id": response.graph_id,
        "embedded_nodes": response.embedded_nodes,
        "skipped_nodes": response.skipped_nodes,
        "provider": response.provider,
        "model": response.model,
        "schema_version": response.schema_version,
        "duration_ms": response.duration_ms,
        "warnings": response.warnings,
    }
