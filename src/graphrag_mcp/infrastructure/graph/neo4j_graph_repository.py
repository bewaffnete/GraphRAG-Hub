"""Neo4j-backed graph repository."""

from __future__ import annotations

from graphrag_mcp.application.ports.parser_port import ParsedGraph
from graphrag_mcp.domain.exceptions import StorageError
from graphrag_mcp.domain.services.scoring_policy import ScoringPolicy
from graphrag_mcp.domain.value_objects.candidate_match import CandidateMatch
from graphrag_mcp.infrastructure.config.settings import Neo4jSettings
from graphrag_mcp.infrastructure.graph.neo4j_mappers import map_graph_edge, map_graph_node
from graphrag_mcp.infrastructure.graph.neo4j_query_builder import build_edge_upsert_query, build_node_upsert_query
from graphrag_mcp.infrastructure.graph.neo4j_schema_manager import Neo4jSchemaManager


class Neo4jGraphRepository:
    def __init__(self, settings: Neo4jSettings, schema_manager: Neo4jSchemaManager | None = None) -> None:
        self._settings = settings
        self._schema_manager = schema_manager or Neo4jSchemaManager()
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise StorageError(
                code="STORAGE_ERROR",
                message="The neo4j package is not installed.",
                hint="Install project dependencies before using the Neo4j backend.",
            ) from exc
        self._driver = GraphDatabase.driver(
            self._settings.uri,
            auth=(self._settings.username, self._settings.password),
        )

    def close(self) -> None:
        self._driver.close()

    def store_library_graph(self, parsed_graph: ParsedGraph) -> None:
        graph_id = parsed_graph.nodes[0].graph_id if parsed_graph.nodes else f"{parsed_graph.library_name}:{parsed_graph.version}"
        embedding_dimensions = next(
            (
                int(node.metadata["embedding_dimensions"])
                for node in parsed_graph.nodes
                if "embedding_dimensions" in node.metadata
            ),
            None,
        )
        try:
            with self._driver.session(database=self._settings.database) as session:
                self._schema_manager.ensure_schema(
                    session,
                    vector_index_name=self._settings.vector_index_name,
                    embedding_dimensions=embedding_dimensions,
                    fulltext_index_name=self._settings.fulltext_index_name,
                )
                session.run("MATCH (n:GraphNode {graph_id: $graph_id}) DETACH DELETE n", graph_id=graph_id)
                for node in parsed_graph.nodes:
                    session.run(
                        build_node_upsert_query(node.kind),
                        node_id=node.node_id,
                        properties=map_graph_node(node),
                    )
                for edge in parsed_graph.edges:
                    session.run(
                        build_edge_upsert_query(edge.type),
                        **map_graph_edge(edge),
                    )
        except Exception as exc:
            raise StorageError(
                code="STORAGE_ERROR",
                message=f"Failed to persist graph {graph_id} to Neo4j: {exc}",
                hint="Verify Neo4j connection settings and schema permissions.",
            ) from exc

    def search_candidates(
        self,
        *,
        query: str,
        graph_ids: list[str] | None,
        labels: list[str] | None,
        top_k: int,
        query_embedding: list[float] | None,
    ) -> list[CandidateMatch]:
        try:
            with self._driver.session(database=self._settings.database) as session:
                records = session.run(
                    """
                    MATCH (n:GraphNode)
                    WHERE n.kind <> 'Library'
                      AND ($graph_ids IS NULL OR n.graph_id IN $graph_ids)
                      AND ($labels IS NULL OR n.kind IN $labels)
                    RETURN n
                    """,
                    graph_ids=graph_ids,
                    labels=labels,
                )
                candidates = [record["n"] for record in records]
                keyword_candidates: dict[str, float] = {}
                try:
                    keyword_records = session.run(
                        """
                        CALL db.index.fulltext.queryNodes($index_name, $query)
                        YIELD node, score
                        WHERE node.kind <> 'Library'
                          AND ($graph_ids IS NULL OR node.graph_id IN $graph_ids)
                          AND ($labels IS NULL OR node.kind IN $labels)
                        RETURN node.node_id AS node_id, score
                        """,
                        index_name=self._settings.fulltext_index_name,
                        query=query,
                        graph_ids=graph_ids,
                        labels=labels,
                    )
                    keyword_candidates = {str(record["node_id"]): float(record["score"]) for record in keyword_records}
                except Exception:
                    keyword_candidates = {}
                vector_candidates: dict[str, float] = {}
                if query_embedding:
                    try:
                        vector_records = session.run(
                            _build_search_clause_query(self._settings.vector_index_name),
                            k=max(top_k * 4, 20),
                            query_embedding=query_embedding,
                            graph_ids=graph_ids,
                            labels=labels,
                        )
                        vector_candidates = {str(record["node_id"]): float(record["score"]) for record in vector_records}
                    except Exception:
                        try:
                            vector_records = session.run(
                                """
                                CALL db.index.vector.queryNodes($index_name, $k, $query_embedding)
                                YIELD node, score
                                WHERE node.kind <> 'Library'
                                  AND ($graph_ids IS NULL OR node.graph_id IN $graph_ids)
                                  AND ($labels IS NULL OR node.kind IN $labels)
                                RETURN node.node_id AS node_id, score
                                """,
                                index_name=self._settings.vector_index_name,
                                k=max(top_k * 4, 20),
                                query_embedding=query_embedding,
                                graph_ids=graph_ids,
                                labels=labels,
                            )
                            vector_candidates = {str(record["node_id"]): float(record["score"]) for record in vector_records}
                        except Exception:
                            vector_candidates = {}
        except Exception as exc:
            raise StorageError(
                code="STORAGE_ERROR",
                message=f"Failed to search candidates in Neo4j: {exc}",
                hint="Verify Neo4j availability before retrieval.",
            ) from exc

        terms = [term for term in query.lower().split() if term]
        scoring = ScoringPolicy()
        matches: list[CandidateMatch] = []
        for node in candidates:
            haystack = " ".join(
                [
                    str(node.get("name", "")),
                    str(node.get("qualified_name", "")),
                    str(node.get("summary", "")),
                    str(node.get("docstring", "")),
                    str(node.get("signature", "")),
                    str(node.get("embedding_text", "")),
                ]
            ).lower()
            keyword_hits = sum(1 for term in terms if term in haystack)
            exact_match = query.lower() in {str(node.get("name", "")).lower(), str(node.get("qualified_name", "")).lower()}
            vector_score = vector_candidates.get(str(node["node_id"]), 0.0)
            bm25_score = keyword_candidates.get(str(node["node_id"]), 0.0)
            if keyword_hits == 0 and bm25_score == 0.0 and vector_score == 0.0 and not exact_match:
                continue
            keyword_score = bm25_score if bm25_score > 0.0 else keyword_hits / max(len(terms), 1)
            score = scoring.score(
                keyword_score=keyword_score,
                vector_score=vector_score,
                is_public=bool(node.get("is_public", True)),
                kind=str(node.get("kind", "Unknown")),
                exact_match=exact_match,
            )
            matches.append(
                CandidateMatch(
                    node_id=str(node["node_id"]),
                    graph_id=str(node["graph_id"]),
                    kind=str(node["kind"]),
                    name=str(node["name"]),
                    qualified_name=str(node["qualified_name"]),
                    summary=str(node.get("summary", "")),
                    score=score,
                    source_path=str(node.get("source_path", "")),
                    keyword_score=keyword_score,
                    vector_score=vector_score,
                )
            )
        matches.sort(key=lambda item: (item.score, item.keyword_score, item.vector_score, item.qualified_name), reverse=True)
        return matches[:top_k]

    def list_graphs(self, *, name_prefix: str | None, status: str | None, limit: int) -> list[dict[str, object]]:
        try:
            with self._driver.session(database=self._settings.database) as session:
                records = session.run(
                    """
                    MATCH (n:GraphNode:Library)
                    WHERE ($name_prefix IS NULL OR toLower(n.name) STARTS WITH toLower($name_prefix))
                      AND ($status IS NULL OR n.status = $status)
                    RETURN n
                    ORDER BY n.updated_at DESC
                    LIMIT $limit
                    """,
                    name_prefix=name_prefix,
                    status=status,
                    limit=limit,
                )
                graphs = []
                for record in records:
                    node = record["n"]
                    graph_id = str(node["graph_id"])
                    count_record = session.run(
                        """
                        MATCH (m:GraphNode {graph_id: $graph_id})
                        RETURN
                          count(CASE WHEN m.kind = 'Module' THEN 1 END) AS modules,
                          count(CASE WHEN m.kind = 'Class' THEN 1 END) AS classes,
                          count(CASE WHEN m.kind = 'Function' THEN 1 END) AS functions,
                          count(CASE WHEN m.kind = 'Example' THEN 1 END) AS examples
                        """,
                        graph_id=graph_id,
                    ).single()
                    graphs.append(
                        {
                            "graph_id": graph_id,
                            "name": str(node["name"]),
                            "version": str(node.get("version", "unknown")),
                            "status": str(node.get("status", "active")),
                            "node_counts": {
                                "modules": int(count_record["modules"]),
                                "classes": int(count_record["classes"]),
                                "functions": int(count_record["functions"]),
                                "examples": int(count_record["examples"]),
                            },
                            "embedding_status": {
                                "ready": bool(node.get("embedding_provider")),
                                "provider": node.get("embedding_provider"),
                                "model": node.get("embedding_model"),
                                "schema_version": node.get("embedding_schema_version"),
                            },
                            "updated_at": node.get("updated_at"),
                        }
                    )
                return graphs
        except Exception as exc:
            raise StorageError(
                code="STORAGE_ERROR",
                message=f"Failed to list graphs in Neo4j: {exc}",
                hint="Verify Neo4j availability before listing graphs.",
            ) from exc

    def get_nodes_by_ids(self, node_ids: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
        requested = set(node_ids)
        try:
            with self._driver.session(database=self._settings.database) as session:
                node_records = session.run(
                    """
                    MATCH (n:GraphNode)
                    WHERE n.node_id IN $node_ids
                    RETURN n
                    """,
                    node_ids=node_ids,
                )
                nodes = []
                found_ids = set()
                for record in node_records:
                    node = record["n"]
                    found_ids.add(str(node["node_id"]))
                    metadata = dict(node)
                    for key in ["node_id", "graph_id", "kind", "name", "qualified_name", "signature", "docstring", "logic_skeleton", "source_excerpt"]:
                        metadata.pop(key, None)
                    nodes.append(
                        {
                            "node_id": str(node["node_id"]),
                            "graph_id": str(node["graph_id"]),
                            "kind": str(node["kind"]),
                            "name": str(node["name"]),
                            "qualified_name": str(node["qualified_name"]),
                            "signature": node.get("signature"),
                            "docstring": node.get("docstring"),
                            "logic_skeleton": node.get("logic_skeleton"),
                            "source_excerpt": node.get("source_excerpt"),
                            "metadata": metadata,
                        }
                    )
                edge_records = session.run(
                    """
                    MATCH (a:GraphNode)-[r]->(b:GraphNode)
                    WHERE a.node_id IN $node_ids AND b.node_id IN $node_ids
                    RETURN a.node_id AS source_node_id, b.node_id AS target_node_id, type(r) AS type
                    """,
                    node_ids=node_ids,
                )
                edges = [record.data() for record in edge_records]
                missing = [node_id for node_id in node_ids if node_id not in found_ids]
                return nodes, edges, missing
        except Exception as exc:
            raise StorageError(
                code="STORAGE_ERROR",
                message=f"Failed to retrieve node details from Neo4j: {exc}",
                hint="Verify Neo4j availability before detail retrieval.",
            ) from exc

    def rebuild_embeddings(self, *, graph_id: str, embedding_provider) -> tuple[int, int]:
        try:
            with self._driver.session(database=self._settings.database) as session:
                records = session.run(
                    """
                    MATCH (n:GraphNode)
                    WHERE n.graph_id = $graph_id
                      AND n.kind IN ['Module', 'Class', 'Function', 'Example']
                    RETURN n
                    """,
                    graph_id=graph_id,
                )
                candidates = [record["n"] for record in records]
                embedded = 0
                for node in candidates:
                    embedding_text = " | ".join(
                        [
                            str(node.get("kind", "")),
                            str(node.get("qualified_name", "")),
                            str(node.get("summary", "")),
                            str(node.get("signature", "")),
                            str(node.get("docstring", "")),
                        ]
                    ).strip(" |")
                    vector = embedding_provider.embed_text(embedding_text)
                    session.run(
                        """
                        MATCH (n:GraphNode {node_id: $node_id})
                        SET n.embedding = $embedding,
                            n.embedding_provider = $provider,
                            n.embedding_model = $model,
                            n.embedding_dimensions = $dimensions,
                            n.embedding_schema_version = $schema_version,
                            n.embedding_text = $embedding_text
                        """,
                        node_id=str(node["node_id"]),
                        embedding=vector,
                        provider=embedding_provider.provider_name,
                        model=embedding_provider.model_name,
                        dimensions=len(vector),
                        schema_version=getattr(embedding_provider, "schema_version", None),
                        embedding_text=embedding_text,
                    )
                    embedded += 1
                return embedded, 0
        except Exception as exc:
            raise StorageError(
                code="STORAGE_ERROR",
                message=f"Failed to rebuild embeddings in Neo4j: {exc}",
                hint="Verify Neo4j and embedding provider availability.",
            ) from exc


def _build_search_clause_query(index_name: str) -> str:
    escaped_name = index_name.replace("`", "``")
    return f"""
    MATCH (node:GraphNode)
      SEARCH node IN (
        VECTOR INDEX `{escaped_name}`
        FOR $query_embedding
        LIMIT $k
      ) SCORE AS score
    WITH node, score
    WHERE node.kind <> 'Library'
      AND ($graph_ids IS NULL OR node.graph_id IN $graph_ids)
      AND ($labels IS NULL OR node.kind IN $labels)
    RETURN node.node_id AS node_id, score
    """
