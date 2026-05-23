"""Implementation logic for MCP tools, bridging to Graph RAG core functionality."""

import os
from types import SimpleNamespace
import yaml

from .schemas import RetrieveInput, ListGraphsInput

from graph_rag.agent_workflow import build_agent_graph
from graph_rag.cli import build_embedding_config, build_neo4j_config
from graph_rag.env import load_app_env
from graph_rag.neo4j_loader import Neo4jConfig
from graph_rag.retriever import GraphDatabase
from graph_rag.retriever import Neo4jGraphRetriever, RetrievalConfig, _public_node_payload
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
    """Execute direct deterministic retrieval, delegating intelligence to the caller."""
    direct_result = _execute_direct_retrieve(input_data)
    if direct_result is None:
        return [{
            "query": input_data.query,
            "answer": "No relevant nodes found.",
            "context": "",
            "sources": [],
            "top_matches": [],
            "candidate_count": 0,
            "selected_count": 0,
            "graph_id": input_data.graph_id,
        }]
    return [direct_result]


def _execute_direct_retrieve(input_data: RetrieveInput) -> dict | None:
    """Fallback to deterministic retrieval when agentic discovery returns nothing."""
    args = _build_runtime_args()
    neo4j_config = build_neo4j_config(args)
    embedding_config = build_embedding_config(args, query_mode=True)
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
    retriever = Neo4jGraphRetriever(neo4j_config, embedding_config)
    try:
        retrieved = retriever.retrieve(input_data.query, retrieval_config)
    except Exception as e:
        print(f"Direct retrieval fallback failed: {e}")
        return None
    finally:
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
    answer = (
        "Retrieved relevant implementation context directly from the graph."
        if retrieved.seeds
        else "No relevant nodes found."
    )
    return {
        "final_answer": answer,
        "retrieved_contexts": [retrieved.compressed_context],
        "selected_ids": selected_ids,
        "candidates": [
            {
                "id": node_payload["graph_id"],
                "name": node_payload["name"],
                "type": node_payload["labels"][0] if node_payload["labels"] else "Unknown",
                "summary": str(node_payload["properties"].get("docstring") or "").strip(),
            }
            for node_payload in (_public_node_payload(node) for node in retrieved.seeds)
        ],
    }


def _build_runtime_args() -> SimpleNamespace:
    """Create a minimal args namespace so MCP uses the same env-driven config as the CLI."""
    provider = os.getenv("EMBEDDING_PROVIDER", "hash")
    return SimpleNamespace(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        provider=provider,
        model=os.getenv("EMBEDDING_MODEL", "hash-embedding-v1"),
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "256")),
        similarity="cosine",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
        gemini_task_type=os.getenv("GEMINI_QUERY_TASK_TYPE", "RETRIEVAL_QUERY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_truncate=True,
        skip_modules=False,
        skip_classes=False,
        skip_functions=False,
        skip_examples=False,
    )


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
