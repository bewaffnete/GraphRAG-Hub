"""Neo4j schema setup for the Graph RAG store."""


class Neo4jSchemaManager:
    def ensure_schema(
        self,
        session,
        *,
        vector_index_name: str | None = None,
        embedding_dimensions: int | None = None,
        fulltext_index_name: str | None = None,
    ) -> None:
        self._drop_legacy_graph_id_constraints(session)
        session.run(
            "CREATE CONSTRAINT graph_node_id IF NOT EXISTS "
            "FOR (n:GraphNode) REQUIRE n.node_id IS UNIQUE"
        )
        session.run(
            "CREATE INDEX graph_node_graph_id IF NOT EXISTS "
            "FOR (n:GraphNode) ON (n.graph_id)"
        )
        session.run(
            "CREATE INDEX graph_node_kind IF NOT EXISTS "
            "FOR (n:GraphNode) ON (n.kind)"
        )
        session.run(
            "CREATE INDEX graph_node_qualified_name IF NOT EXISTS "
            "FOR (n:GraphNode) ON (n.qualified_name)"
        )
        if fulltext_index_name:
            session.run(
                f"""
                CREATE FULLTEXT INDEX {fulltext_index_name} IF NOT EXISTS
                FOR (n:GraphNode)
                ON EACH [n.name, n.qualified_name, n.summary, n.docstring, n.signature, n.embedding_text]
                """
            )
        if vector_index_name and embedding_dimensions:
            session.run(
                f"""
                CREATE VECTOR INDEX {vector_index_name} IF NOT EXISTS
                FOR (n:GraphNode) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {embedding_dimensions},
                  `vector.similarity_function`: 'cosine'
                }}}}
                """
            )

    def _drop_legacy_graph_id_constraints(self, session) -> None:
        records = session.run(
            "SHOW CONSTRAINTS "
            "YIELD name, labelsOrTypes, properties "
            "RETURN name, labelsOrTypes, properties"
        )
        for record in records:
            labels = record.get("labelsOrTypes") or []
            properties = record.get("properties") or []
            if "graph_id" not in properties:
                continue
            if "GraphNode" in labels:
                continue
            name = record.get("name")
            if name:
                escaped_name = str(name).replace("`", "``")
                session.run(f"DROP CONSTRAINT `{escaped_name}` IF EXISTS")
