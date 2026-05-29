"""Generic graph node entity."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class GraphNode:
    node_id: str
    graph_id: str
    kind: str
    name: str
    display_name: str
    qualified_name: str
    source_path: str
    line_start: int | None
    line_end: int | None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    summary: str | None = None
    docstring: str | None = None
    signature: str | None = None
    logic_skeleton: str | None = None
    source_excerpt: str | None = None
    is_public: bool = True
    metadata: dict[str, object] = field(default_factory=dict)
