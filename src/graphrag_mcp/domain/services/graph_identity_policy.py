"""Stable graph identity helpers."""

from graphrag_mcp.domain.value_objects.graph_id import GraphId
from graphrag_mcp.domain.value_objects.node_id import NodeId


class GraphIdentityPolicy:
    def build_graph_id(self, library_name: str, version: str) -> str:
        return str(GraphId.create(library_name=library_name, version=version))

    def build_node_id(self, graph_id: str, kind: str, qualified_identity: str) -> str:
        library_name, version = graph_id.split(":", 1)
        return str(
            NodeId(
                graph_id=GraphId(library_name=library_name, version=version),
                kind=kind,
                qualified_identity=qualified_identity,
            )
        )
