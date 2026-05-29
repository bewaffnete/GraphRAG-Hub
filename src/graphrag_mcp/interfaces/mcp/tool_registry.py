"""Minimal MCP tool registry."""

from graphrag_mcp.interfaces.mcp.handlers.health_check import handle_graphrag_health
from graphrag_mcp.interfaces.mcp.handlers.ingest_library import handle_graphrag_ingest_library
from graphrag_mcp.interfaces.mcp.handlers.list_graphs import handle_graphrag_list_graphs
from graphrag_mcp.interfaces.mcp.handlers.rebuild_embeddings import handle_graphrag_rebuild_embeddings
from graphrag_mcp.interfaces.mcp.handlers.retrieve_candidates import handle_graphrag_retrieve_candidates
from graphrag_mcp.interfaces.mcp.handlers.retrieve_details import handle_graphrag_retrieve_details


def build_tool_registry(container) -> dict[str, callable]:
    return {
        "graphrag_ingest_library": lambda payload: handle_graphrag_ingest_library(payload, container.ingest_library),
        "graphrag_list_graphs": lambda payload: handle_graphrag_list_graphs(payload, container.list_graphs),
        "graphrag_retrieve_candidates": lambda payload: handle_graphrag_retrieve_candidates(payload, container.retrieve_candidates),
        "graphrag_retrieve_details": lambda payload: handle_graphrag_retrieve_details(payload, container.retrieve_details),
        "graphrag_rebuild_embeddings": lambda payload: handle_graphrag_rebuild_embeddings(payload, container.rebuild_embeddings),
        "graphrag_health": lambda payload: handle_graphrag_health(payload, container.health_check),
    }
