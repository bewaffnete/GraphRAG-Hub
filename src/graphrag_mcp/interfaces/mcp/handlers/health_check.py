"""MCP health handler."""

from graphrag_mcp.application.use_cases.health_check import HealthCheckUseCase
from graphrag_mcp.interfaces.mcp.handlers.error_mapping import map_error


def handle_graphrag_health(_payload: dict[str, object], use_case: HealthCheckUseCase) -> dict[str, object]:
    try:
        response = use_case.execute()
    except Exception as exc:
        return map_error(exc)
    return {
        "status": response.status,
        "neo4j": {"status": response.neo4j.status, "message": response.neo4j.message, **response.neo4j.details},
        "registry": {"status": response.registry.status, "message": response.registry.message, **response.registry.details},
        "embedding_provider": {"status": response.embedding_provider.status, "message": response.embedding_provider.message, **response.embedding_provider.details},
        "server_version": response.server_version,
        "warnings": response.warnings,
    }
