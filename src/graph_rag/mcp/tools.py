import os
from pathlib import Path
import yaml
import argparse

from .schemas import RetrieveInput, ChatInput, ListGraphsInput, IngestInput

from graph_rag.retriever import Neo4jGraphRetriever, RetrievalConfig
from graph_rag.cli import build_neo4j_config, build_embedding_config, load_snapshot_to_neo4j, embed_graph
from graph_rag.parser import parse_python_library
from graph_rag.agent_workflow import build_agent_graph
from graph_rag.query_decomposition import register_graph_in_config


def get_mock_args() -> argparse.Namespace:
    return argparse.Namespace(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        provider=os.getenv("EMBEDDING_PROVIDER", "hash"),
        model=os.getenv("EMBEDDING_MODEL", "hash-embedding-v1"),
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "256")),
        similarity="cosine",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
        gemini_task_type=os.getenv("GEMINI_TASK_TYPE", "RETRIEVAL_DOCUMENT"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_truncate=True,
    )


async def execute_retrieve(input_data: RetrieveInput) -> list[dict]:
    args = get_mock_args()
    args.gemini_task_type = os.getenv("GEMINI_QUERY_TASK_TYPE", "RETRIEVAL_QUERY")
    
    neo4j_config = build_neo4j_config(args)
    embedding_config = build_embedding_config(args, query_mode=True)
    
    retrieval_config = RetrievalConfig(
        graph_id=input_data.graph_id,
        top_k=input_data.top_k,
        max_entities=input_data.top_k,
    )
    
    retriever = Neo4jGraphRetriever(neo4j_config, embedding_config)
    try:
        result = retriever.retrieve(input_data.query, retrieval_config)
        
        # Return list of ContextNode dicts
        # We can extract from result.nodes
        context_nodes = []
        for node in result.nodes:
            context_nodes.append({
                "name": node.name,
                "type": node.labels[0] if node.labels else "Unknown",
                "docstring": node.properties.get("docstring"),
                "signature": node.properties.get("signature"),
                "graph_id": node.graph_id,
            })
        return context_nodes
    finally:
        retriever.close()


async def execute_chat(input_data: ChatInput) -> dict:
    app = build_agent_graph()
    # The agent might use process env vars natively
    initial_state = {
        "query": input_data.query,
        "sub_query_results": [],
        "final_answer": "",
        "candidates": [],
        "selected_ids": [],
        "retrieved_contexts": [],
        "decomposition": None
    }
    
    result = app.invoke(initial_state)
    return {
        "answer": result.get("final_answer", "No answer generated."),
        "sources": result.get("selected_ids", [])
    }


async def execute_list_graphs(input_data: ListGraphsInput) -> list[dict]:
    yaml_path = Path("available_graphs.yaml")
    if not yaml_path.exists():
        return []
    
    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not content or "graphs" not in content:
            return []
        
        graphs = []
        for g in content.get("graphs", []):
            graphs.append({
                "name": g.get("name", ""),
                "versions": g.get("versions", []),
                "latest": g.get("latest", ""),
                "status": g.get("status", "")
            })
        return graphs
    except Exception as e:
        print(f"Failed to read available_graphs.yaml: {e}")
        return []


async def execute_ingest(input_data: IngestInput) -> dict:
    args = get_mock_args()
    args.provider = input_data.provider
    args.batch_size = 32
    args.batch_delay_seconds = None
    args.max_text_length = 12000
    args.skip_modules = False
    args.skip_classes = False
    args.skip_functions = False
    args.skip_examples = False
    args.embedding_schema_version = "v1"
    
    # 1. Parse
    snapshot = parse_python_library(Path(input_data.source))
    
    # 2. Normalize and identify graph_id
    if input_data.graph_id and ":" in input_data.graph_id:
        lib_name, version = input_data.graph_id.split(":", 1)
        snapshot.metadata.name = lib_name.replace("_", "-").lower()
        if version and version != "latest":
            snapshot.metadata.version = version
    elif input_data.graph_id:
        snapshot.metadata.name = input_data.graph_id.replace("_", "-").lower()

    # 3. Load
    load_result = load_snapshot_to_neo4j(snapshot, args)
    
    # Register in available_graphs
    register_graph_in_config(snapshot.metadata.name, snapshot.metadata.version or "unknown")
    
    # 4. Embed
    embed_result = embed_graph(load_result["graph_id"], args)
    
    return {
        "status": "ok",
        "nodes_created": load_result.get("nodes_created", 0),
        "graph_id": load_result.get("graph_id")
    }

