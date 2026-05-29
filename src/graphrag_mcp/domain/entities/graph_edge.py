"""Graph edge entity."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class GraphEdge:
    source_node_id: str
    target_node_id: str
    type: str
    properties: dict[str, object] = field(default_factory=dict)
