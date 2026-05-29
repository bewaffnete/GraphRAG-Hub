"""Cypher helpers for graph persistence."""

from graphrag_mcp.domain.exceptions import StorageError

ALLOWED_NODE_KINDS = {
    "Library",
    "Module",
    "Class",
    "Function",
    "Parameter",
    "ReturnType",
    "Exception",
    "Example",
}

ALLOWED_EDGE_TYPES = {
    "CONTAINS",
    "DECLARES",
    "HAS_METHOD",
    "HAS_PARAM",
    "RETURNS",
    "RAISES",
    "CALLS",
    "INHERITS",
    "USES_EXAMPLE",
}


def build_node_upsert_query(kind: str) -> str:
    if kind not in ALLOWED_NODE_KINDS:
        raise StorageError(
            code="STORAGE_ERROR",
            message=f"Unsupported node kind for Neo4j persistence: {kind}",
            hint="Keep node kinds aligned with SCHEMA.md.",
        )
    return f"MERGE (n:GraphNode:{kind} {{node_id: $node_id}}) SET n += $properties"


def build_edge_upsert_query(edge_type: str) -> str:
    if edge_type not in ALLOWED_EDGE_TYPES:
        raise StorageError(
            code="STORAGE_ERROR",
            message=f"Unsupported edge type for Neo4j persistence: {edge_type}",
            hint="Keep edge types aligned with SCHEMA.md.",
        )
    return (
        "MATCH (source:GraphNode {node_id: $source_node_id}) "
        "MATCH (target:GraphNode {node_id: $target_node_id}) "
        f"MERGE (source)-[rel:{edge_type}]->(target) "
        "SET rel += $properties"
    )
