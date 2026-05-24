"""Implementation logic for MCP tools, bridging to Graph RAG core functionality."""

import os
import yaml

from .schemas import RetrieveInput, ListGraphsInput

from graph_rag.embedding_indexer import EmbeddingConfig
from graph_rag.env import load_app_env
from graph_rag.neo4j_loader import Neo4jConfig
from graph_rag.retriever import GraphDatabase
from graph_rag.retriever import Neo4jGraphRetriever, RetrievalConfig
from graph_rag.paths import resolve_available_graphs_path

load_app_env()


def prioritize_top_matches(matches: list[dict]) -> list[dict]:
    """Sort and deduplicate selected matches for cleaner MCP output."""
    priority = {
        "Function": 0,
        "Class": 1,
        "Module": 2,
        "Example": 3,
    }
    unique: dict[str, dict] = {}
    for match in matches:
        match_id = str(match.get("id") or "").strip()
        if not match_id or match_id in unique:
            continue
        unique[match_id] = match
    return sorted(
        unique.values(),
        key=lambda item: (
            priority.get(str(item.get("type") or "Unknown"), 9),
            str(item.get("name") or ""),
        ),
    )


def dedupe_ids(values: list[str]) -> list[str]:
    """Deduplicate ids while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


async def execute_retrieve(input_data: RetrieveInput) -> list[dict]:
    """Execute deterministic retrieval and return graph evidence for the caller to reason over."""
    direct_result = _execute_direct_retrieve(input_data)
    if direct_result is None:
        return [{
            "query": input_data.query,
            "answer": "No graph evidence could be retrieved.",
            "context": "",
            "sources": [],
            "top_matches": [],
            "candidate_count": 0,
            "selected_count": 0,
            "graph_id": input_data.graph_id,
            "routed_graphs": [],
        }]
    return [direct_result]


def _execute_direct_retrieve(input_data: RetrieveInput) -> dict | None:
    """Run graph retrieval without invoking an internal LLM."""
    neo4j_config = _build_neo4j_config()
    embedding_config = _build_embedding_config()
    retrieval_config = RetrievalConfig(
        graph_id=input_data.graph_id,
        top_k=max(input_data.top_k, 1),
        keyword_k=max(input_data.top_k, 8),
        vector_k=max(input_data.top_k, 8),
        hops=2,
        route_top_k=3,
        context_max_chars=12000,
        max_entities=max(input_data.top_k, 5),
        class_boost=1.5,
        public_boost=1.25,
        private_penalty=0.45,
    )
    retriever = None
    try:
        retriever = Neo4jGraphRetriever(neo4j_config, embedding_config)
        retrieved = retriever.retrieve(input_data.query, retrieval_config)
    except Exception as e:
        print(f"Direct retrieval failed: {e}")
        return None
    finally:
        if retriever is not None:
            retriever.close()

    top_matches = prioritize_top_matches(
        [
            {
                "id": node.graph_id,
                "name": node.name,
                "type": node.labels[0] if node.labels else "Unknown",
                "summary": str(node.properties.get("docstring") or "").strip(),
            }
            for node in retrieved.seeds
        ]
    )
    selected_ids = dedupe_ids([node.graph_id for node in retrieved.nodes])
    summary = (
        "Retrieved deterministic graph evidence. Use top_matches and context to choose the useful nodes."
        if retrieved.seeds
        else "No relevant nodes found."
    )
    return {
        "query": input_data.query,
        "answer": summary,
        "context": retrieved.compressed_context,
        "sources": selected_ids,
        "top_matches": top_matches,
        "candidate_count": len(retrieved.seeds),
        "selected_count": len(selected_ids),
        "graph_id": retrieved.graph_id,
        "routed_graphs": retrieved.routed_graphs,
    }


def _build_neo4j_config() -> Neo4jConfig:
    """Create Neo4j config directly from the MCP process environment."""
    return Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        create_vector_indexes=True,
        vector_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "256")),
        vector_similarity=os.getenv("EMBEDDING_SIMILARITY", "cosine"),
    )


def _build_embedding_config() -> EmbeddingConfig:
    """Create embedding config directly from the MCP process environment."""
    provider = os.getenv("EMBEDDING_PROVIDER", "hash")
    model = _resolve_embedding_model(provider, os.getenv("EMBEDDING_MODEL", "hash-embedding-v1"))
    return EmbeddingConfig(
        provider=provider,
        model=model,
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "256")),
        similarity=os.getenv("EMBEDDING_SIMILARITY", "cosine"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
        gemini_task_type=os.getenv("GEMINI_QUERY_TASK_TYPE", "RETRIEVAL_QUERY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_truncate=True,
    )


def _resolve_embedding_model(provider: str, model: str) -> str:
    """Keep MCP defaults aligned with CLI embedding defaults without importing CLI."""
    if model != "hash-embedding-v1":
        return model
    if provider == "gemini":
        return "gemini-embedding-001"
    if provider == "openai":
        return "text-embedding-3-large"
    if provider == "ollama":
        return "embeddinggemma"
    return model


async def execute_list_graphs(input_data: ListGraphsInput) -> list[dict]:
    """Fetch indexed libraries, preferring Neo4j and falling back to the YAML registry."""
    neo4j_graphs = _list_graphs_from_neo4j()
    if neo4j_graphs:
        return neo4j_graphs

    yaml_path = resolve_available_graphs_path()
    if not yaml_path.exists():
        return []
    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not content or "graphs" not in content:
            return []

        graphs = []
        for g in content.get("graphs", []):
            name = g.get("name", "")
            versions = g.get("versions", []) or []
            graph_ids = [f"{name}:{version}" for version in versions if name and version]
            graphs.append({
                "name": name,
                "latest": g.get("latest", ""),
                "graph_ids": graph_ids,
            })
        return graphs
    except Exception as e:
        print(f"Failed to read available_graphs.yaml: {e}")
        return []


def _list_graphs_from_neo4j() -> list[dict]:
    """Return graph metadata directly from Neo4j when the database is reachable."""
    if GraphDatabase is None:
        return []

    config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    if not config.password:
        return []

    driver = GraphDatabase.driver(
        config.uri,
        auth=(config.username, config.password),
        notifications_min_severity="OFF",
    )
    try:
        with driver.session(database=config.database) as session:
            result = session.run(
                """
                MATCH (n:Library)
                WITH n.name AS name, n.version AS version
                WHERE name IS NOT NULL AND version IS NOT NULL
                WITH name, collect(DISTINCT version) AS versions
                RETURN name, versions
                ORDER BY toLower(name)
                """
            )
            graphs = []
            for record in result:
                name = record["name"]
                versions = sorted(record["versions"])
                graphs.append(
                    {
                        "name": name,
                        "latest": versions[-1] if versions else "",
                        "graph_ids": [f"{name}:{version}" for version in versions],
                    }
                )
            return graphs
    except Exception as e:
        print(f"Failed to list graphs from Neo4j: {e}")
        return []
    finally:
        driver.close()
