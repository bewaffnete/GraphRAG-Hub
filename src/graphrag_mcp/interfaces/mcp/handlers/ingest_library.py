"""MCP ingest handler."""

from graphrag_mcp.application.dto.ingest import IngestLibraryRequest
from graphrag_mcp.application.use_cases.ingest_library import IngestLibraryUseCase
from graphrag_mcp.interfaces.mcp.handlers.error_mapping import map_error


def handle_graphrag_ingest_library(payload: dict[str, object], use_case: IngestLibraryUseCase) -> dict[str, object]:
    try:
        request = IngestLibraryRequest(
            path=str(payload["path"]),
            library_name=str(payload["library_name"]) if payload.get("library_name") is not None else None,
            version=str(payload["version"]) if payload.get("version") is not None else None,
            ingest_mode=str(payload.get("ingest_mode", "parse_load_embed")),
            embedding_mode=str(payload.get("embedding_mode", "enabled")),
        )
        response = use_case.execute(request)
    except Exception as exc:
        return map_error(exc)

    return {
        "graph_id": response.graph_id,
        "library_name": response.library_name,
        "version": response.version,
        "counts": response.counts,
        "embedding_summary": {
            "enabled": response.embedding_summary.enabled,
            "embedded_nodes": response.embedding_summary.embedded_nodes,
            "provider": response.embedding_summary.provider,
            "model": response.embedding_summary.model,
        },
        "duration_ms": response.duration_ms,
        "warnings": response.warnings,
        "executed_stages": response.executed_stages,
    }
