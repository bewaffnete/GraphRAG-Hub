"""MCP-safe error envelope helpers."""

from graphrag_mcp.domain.exceptions import GraphragError


def map_error(exc: Exception) -> dict[str, object]:
    if isinstance(exc, GraphragError):
        return {
            "error": {
                "code": exc.code,
                "message": exc.message,
                "hint": exc.hint,
            }
        }
    return {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": str(exc),
            "hint": "Inspect server logs for details.",
        }
    }
