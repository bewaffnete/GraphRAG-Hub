"""Node identity value object."""

from dataclasses import dataclass

from graphrag_mcp.domain.value_objects.graph_id import GraphId


@dataclass(frozen=True, slots=True)
class NodeId:
    graph_id: GraphId
    kind: str
    qualified_identity: str

    def __str__(self) -> str:
        return f"{self.graph_id}:{self.kind}:{self.qualified_identity}"
