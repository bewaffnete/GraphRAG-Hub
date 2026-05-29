"""MCP retrieve details handler."""

from graphrag_mcp.application.dto.retrieve_details import RetrieveDetailsRequest
from graphrag_mcp.application.use_cases.retrieve_details import RetrieveDetailsUseCase
from graphrag_mcp.interfaces.mcp.handlers.error_mapping import map_error


def handle_graphrag_retrieve_details(payload: dict[str, object], use_case: RetrieveDetailsUseCase) -> dict[str, object]:
    try:
        request = RetrieveDetailsRequest(
            node_ids=[str(item) for item in payload.get("node_ids", [])],
            include_relationships=bool(payload.get("include_relationships", True)),
            include_docstring=bool(payload.get("include_docstring", True)),
            include_logic_skeleton=bool(payload.get("include_logic_skeleton", True)),
            include_source_excerpt=bool(payload.get("include_source_excerpt", False)),
        )
        response = use_case.execute(request)
    except Exception as exc:
        return map_error(exc)
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "graph_id": node.graph_id,
                "kind": node.kind,
                "name": node.name,
                "qualified_name": node.qualified_name,
                "signature": node.signature,
                "docstring": node.docstring,
                "logic_skeleton": node.logic_skeleton,
                "source_excerpt": node.source_excerpt,
                # "metadata": node.metadata,
            }
            for node in response.nodes
        ],
        "edges": [{"source_node_id": edge.source_node_id, "target_node_id": edge.target_node_id, "type": edge.type} for edge in response.edges],
        "expanded_context": response.expanded_context,
        "missing_node_ids": response.missing_node_ids,
        "warnings": response.warnings,
    }
