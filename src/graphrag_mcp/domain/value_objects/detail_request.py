"""Detail request value object."""

from dataclasses import dataclass

from graphrag_mcp.domain.exceptions import GraphragError


@dataclass(frozen=True, slots=True)
class DetailRequest:
    node_ids: tuple[str, ...]
    include_relationships: bool = True
    include_docstring: bool = True
    include_logic_skeleton: bool = True
    include_source_excerpt: bool = False

    def __post_init__(self) -> None:
        if not self.node_ids:
            raise GraphragError(
                code="INVALID_INPUT",
                message="node_ids must not be empty.",
                hint="Pass at least one selected node_id.",
            )
        if len(self.node_ids) > 100:
            raise GraphragError(
                code="INVALID_INPUT",
                message="node_ids must not contain more than 100 items.",
                hint="Request details in smaller batches.",
            )
