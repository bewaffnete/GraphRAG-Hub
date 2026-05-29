from graphrag_mcp.infrastructure.graph.neo4j_schema_manager import Neo4jSchemaManager


class _FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)


class _FakeSession:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, query: str, **kwargs):
        self.commands.append(query)
        if query.startswith("SHOW CONSTRAINTS"):
            return _FakeResult(
                [
                    {"name": "function_graph_id", "labelsOrTypes": ["Function"], "properties": ["graph_id"]},
                    {"name": "graph_node_id", "labelsOrTypes": ["GraphNode"], "properties": ["node_id"]},
                ]
            )
        return _FakeResult([])


def test_ensure_schema_drops_legacy_label_graph_id_constraints() -> None:
    session = _FakeSession()

    Neo4jSchemaManager().ensure_schema(session)

    assert "DROP CONSTRAINT `function_graph_id` IF EXISTS" in session.commands
    assert (
        "CREATE CONSTRAINT graph_node_id IF NOT EXISTS "
        "FOR (n:GraphNode) REQUIRE n.node_id IS UNIQUE"
    ) in session.commands


def test_ensure_schema_does_not_drop_graph_node_constraint() -> None:
    session = _FakeSession()

    Neo4jSchemaManager().ensure_schema(session)

    assert "DROP CONSTRAINT `graph_node_id` IF EXISTS" not in session.commands
