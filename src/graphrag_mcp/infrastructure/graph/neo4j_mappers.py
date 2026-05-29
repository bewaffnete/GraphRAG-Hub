"""Neo4j persistence mappers."""

from graphrag_mcp.domain.entities.graph_edge import GraphEdge
from graphrag_mcp.domain.entities.graph_node import GraphNode


def map_graph_node(node: GraphNode) -> dict[str, object]:
    return {
        "node_id": node.node_id,
        "graph_id": node.graph_id,
        "kind": node.kind,
        "name": node.name,
        "display_name": node.display_name,
        "qualified_name": node.qualified_name,
        "source_path": node.source_path,
        "line_start": node.line_start,
        "line_end": node.line_end,
        "created_at": node.created_at,
        "updated_at": node.updated_at,
        "summary": node.summary,
        "docstring": node.docstring,
        "signature": node.signature,
        "logic_skeleton": node.logic_skeleton,
        "source_excerpt": node.source_excerpt,
        "is_public": node.is_public,
        **node.metadata,
    }


def map_graph_edge(edge: GraphEdge) -> dict[str, object]:
    return {
        "source_node_id": edge.source_node_id,
        "target_node_id": edge.target_node_id,
        "properties": dict(edge.properties),
    }
