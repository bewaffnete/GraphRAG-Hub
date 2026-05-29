"""Simple in-memory registry."""


class InMemoryGraphRegistry:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, object]] = {}

    def register_graph(self, graph_id: str, metadata: dict[str, object]) -> None:
        self.items[graph_id] = metadata
