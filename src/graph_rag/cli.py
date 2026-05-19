"""
Command-line interface for the Graph RAG pipeline.

Provides commands for parsing libraries, loading snapshots into Neo4j,
generating embeddings, and performing retrieval/chat operations.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

# Suppress warnings (including LangChain/LangGraph deprecations)
warnings.filterwarnings("ignore")

from .embedding_indexer import EMBEDDING_SCHEMA_VERSION, EmbeddingConfig, Neo4jEmbeddingIndexer
from .models import LibrarySnapshot
from .neo4j_loader import Neo4jConfig, Neo4jGraphLoader
from .parser import parse_python_library
from .retriever import Neo4jGraphRetriever, RetrievalConfig

KNOWN_GEMINI_MODELS = ("gemini-embedding-2", "gemini-embedding-001")
KNOWN_OLLAMA_MODELS = ("embeddinggemma", "qwen3-embedding", "all-minilm")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the Graph RAG CLI."""
    parser = argparse.ArgumentParser(description="Graph RAG pipeline for parsing, loading, embedding, and retrieval.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse a local library/repository into JSON.")
    parse_parser.add_argument("path", help="Path to a local repository or package root.")
    parse_parser.add_argument("--output", "-o", help="Optional output JSON path.")
    parse_parser.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    parse_parser.set_defaults(func=run_parse)

    load_parser = subparsers.add_parser("load", help="Load a parsed snapshot into Neo4j.")
    source_group = load_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--snapshot", help="Path to a parsed snapshot JSON file.")
    source_group.add_argument("--path", help="Path to parse and load directly.")
    add_neo4j_args(load_parser)
    load_parser.set_defaults(func=run_load)

    embed_parser = subparsers.add_parser("embed", help="Generate embeddings for a loaded graph.")
    embed_parser.add_argument("--graph-id", required=True, help="Graph id, e.g. scikit-learn:1.5.2")
    add_neo4j_args(embed_parser)
    add_embedding_args(embed_parser, query_mode=False)
    embed_parser.set_defaults(func=run_embed)

    ingest_parser = subparsers.add_parser("ingest", help="Parse, load into Neo4j, and embed in one command.")
    ingest_parser.add_argument("path", nargs="?", help="Path to a local repository or package root. If omitted, opens an interactive selector.")
    ingest_parser.add_argument("--snapshot-output", help="Optional path to write the parsed snapshot JSON.")
    ingest_parser.add_argument("--skip-embed", action="store_true", help="Only parse and load, without embeddings.")
    add_neo4j_args(ingest_parser)
    add_embedding_args(ingest_parser, query_mode=False)
    ingest_parser.set_defaults(func=run_ingest)

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve relevant implementation context from Neo4j.")
    retrieve_parser.add_argument("query", help="User query for retrieval.")
    retrieve_parser.add_argument("--graph-id", help="Optional graph id to search within.")
    add_neo4j_args(retrieve_parser)
    add_embedding_args(retrieve_parser, query_mode=True)
    retrieve_parser.add_argument("--top-k", type=int, default=10, help="Candidate count before compacting output.")
    retrieve_parser.add_argument("--max-entities", type=int, default=5, help="Max entities to return.")
    retrieve_parser.add_argument("--keyword-k", type=int, default=8, help="Keyword candidate count per label.")
    retrieve_parser.add_argument("--vector-k", type=int, default=8, help="Vector candidate count per label.")
    retrieve_parser.add_argument("--hops", type=int, default=2, choices=[1, 2], help="Traversal depth around matched nodes.")
    retrieve_parser.add_argument("--route-top-k", type=int, default=3, help="Candidate libraries when --graph-id is omitted.")
    retrieve_parser.add_argument("--context-max-chars", type=int, default=12000, help="Max compressed context size.")
    retrieve_parser.add_argument("--class-boost", type=float, default=1.5, help="Score multiplier for Class nodes.")
    retrieve_parser.add_argument("--public-boost", type=float, default=1.25, help="Score multiplier for public API nodes.")
    retrieve_parser.add_argument("--private-penalty", type=float, default=0.45, help="Score multiplier for private/internal nodes.")
    retrieve_parser.add_argument("--json", action="store_true", help="Return JSON instead of text.")
    retrieve_parser.add_argument("--output", "-o", help="Optional output path.")
    retrieve_parser.set_defaults(func=run_retrieve)

    chat_parser = subparsers.add_parser("chat", help="Agentic search with query decomposition and synthesis.")
    chat_parser.add_argument("query", help="User query for the agent.")
    chat_parser.set_defaults(func=run_chat)

    setup_parser = subparsers.add_parser("setup", help="Open the interactive configuration control panel.")
    setup_parser.set_defaults(func=run_setup)

    return parser


def add_neo4j_args(parser: argparse.ArgumentParser) -> None:
    """Add standard Neo4j connection arguments to a parser."""
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j URI.")
    parser.add_argument("--username", default=os.getenv("NEO4J_USERNAME", "neo4j"), help="Neo4j username.")
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD"), help="Neo4j password.")
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"), help="Neo4j database.")


def add_embedding_args(parser: argparse.ArgumentParser, *, query_mode: bool) -> None:
    """Add embedding provider and model arguments to a parser."""
    parser.add_argument("--provider", default="hash", choices=["hash", "openai", "gemini", "ollama"], help="Embedding provider.")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "hash-embedding-v1"), help="Embedding model name.")
    parser.add_argument("--dimensions", type=int, default=int(os.getenv("EMBEDDING_DIMENSIONS", "256")), help="Embedding dimensions.")
    if not query_mode:
        parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
        parser.add_argument("--batch-delay-seconds", type=float, default=None, help="Sleep between embedding batches.")
        parser.add_argument("--max-text-length", type=int, default=12000, help="Max embedded text length.")
        parser.add_argument("--skip-modules", action="store_true", help="Do not embed Module nodes.")
        parser.add_argument("--skip-classes", action="store_true", help="Do not embed Class nodes.")
        parser.add_argument("--skip-functions", action="store_true", help="Do not embed Function nodes.")
        parser.add_argument("--skip-examples", action="store_true", help="Do not embed Example nodes.")
        parser.add_argument("--embedding-schema-version", default=EMBEDDING_SCHEMA_VERSION, help="Embedding schema version used to invalidate stale vectors.")
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key.")
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="OpenAI-compatible base URL.")
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API key.")
    parser.add_argument("--gemini-base-url", default=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"), help="Gemini API base URL.")
    default_gemini_task = os.getenv("GEMINI_QUERY_TASK_TYPE" if query_mode else "GEMINI_TASK_TYPE")
    parser.add_argument("--gemini-task-type", default=default_gemini_task or ("RETRIEVAL_QUERY" if query_mode else "RETRIEVAL_DOCUMENT"), help="Gemini task type.")
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), help="Ollama base URL.")
    if not query_mode:
        parser.add_argument("--ollama-truncate", dest="ollama_truncate", action="store_true", help="Allow Ollama input truncation.")
        parser.add_argument("--no-ollama-truncate", dest="ollama_truncate", action="store_false", help="Disable Ollama input truncation.")
        parser.set_defaults(ollama_truncate=True)
    parser.add_argument("--similarity", default="cosine", choices=["cosine", "euclidean"], help="Vector similarity function.")


def run_parse(args: argparse.Namespace) -> None:
    """Run the 'parse' command."""
    snapshot = parse_python_library(Path(args.path))
    payload = snapshot.to_json(indent=args.indent)
    write_or_print(payload, args.output)


def run_load(args: argparse.Namespace) -> None:
    """Run the 'load' command."""
    from .query_decomposition import register_graph_in_config
    snapshot = load_snapshot_from_args(args)
    result = load_snapshot_to_neo4j(snapshot, args)
    print_load_result(result, args.database)
    name = snapshot.metadata.name.replace("_", "-").lower()
    version = snapshot.metadata.version or "unknown"
    register_graph_in_config(name, version)


def run_embed(args: argparse.Namespace) -> None:
    """Run the 'embed' command."""
    result = embed_graph(args.graph_id, args)
    print(f"Indexed graph_id={result['graph_id']}")
    print(f"Embedded nodes={result['embedded_nodes']}")


def run_ingest(args: argparse.Namespace) -> None:
    """Run the 'ingest' command (parse + load + embed)."""
    from .query_decomposition import register_graph_in_config
    if not args.path:
        from .config_ui import select_installed_package_interactive
        path = select_installed_package_interactive()
        if not path:
            print("No path selected. Exiting.")
            return
        args.path = path

    snapshot = parse_python_library(Path(args.path))
    if args.snapshot_output:
        Path(args.snapshot_output).write_text(snapshot.to_json(), encoding="utf-8")
    load_result = load_snapshot_to_neo4j(snapshot, args)
    print_load_result(load_result, args.database)
    
    name = snapshot.metadata.name.replace("_", "-").lower()
    version = snapshot.metadata.version or "unknown"
    register_graph_in_config(name, version)

    if args.skip_embed:
        return
    embed_result = embed_graph(load_result["graph_id"], args)
    print(f"Indexed graph_id={embed_result['graph_id']}")
    print(f"Embedded nodes={embed_result['embedded_nodes']}")


def run_retrieve(args: argparse.Namespace) -> None:
    """Run the 'retrieve' command."""
    require_neo4j_password(args)
    neo4j_config = build_neo4j_config(args)
    embedding_config = build_embedding_config(args, query_mode=True)
    retrieval_config = RetrievalConfig(
        graph_id=args.graph_id,
        top_k=args.top_k,
        keyword_k=args.keyword_k,
        vector_k=args.vector_k,
        hops=args.hops,
        route_top_k=args.route_top_k,
        context_max_chars=args.context_max_chars,
        max_entities=args.max_entities,
        class_boost=args.class_boost,
        public_boost=args.public_boost,
        private_penalty=args.private_penalty,
    )
    retriever = Neo4jGraphRetriever(neo4j_config, embedding_config)
    try:
        result = retriever.retrieve(args.query, retrieval_config)
    finally:
        retriever.close()

    if args.json:
        write_or_print(result.to_json(), args.output)
        return

    lines = [f"Graph: {result.graph_id}"]
    if result.routed_graphs:
        lines.append("Routed graphs: " + ", ".join(result.routed_graphs))
    lines.extend(["", result.compressed_context])
    write_or_print("\n".join(lines), args.output)


def run_chat(args: argparse.Namespace) -> None:
    """
    Run the 'chat' command (agentic workflow).

    Invokes the LangGraph-based workflow to decompose the query,
    retrieve context from multiple libraries, and synthesize a final answer.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    from .agent_workflow import build_agent_graph
    
    app = build_agent_graph()
    initial_state = {
        "query": args.query,
        "sub_query_results": [],
        "final_answer": ""
    }
    
    result = app.invoke(initial_state)
    print("\n" + "="*40)
    print("FINAL AGENT ANSWER")
    print("="*40)
    print(result.get("final_answer", "No answer generated."))


def run_setup(args: argparse.Namespace) -> None:
    """
    Open the interactive configuration control panel.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    interactive_main()


def load_snapshot_from_args(args: argparse.Namespace) -> LibrarySnapshot:
    """Helper to load a LibrarySnapshot from JSON or by parsing a path."""
    if getattr(args, "snapshot", None):
        return LibrarySnapshot.from_json_file(args.snapshot)
    return parse_python_library(Path(args.path))


def load_snapshot_to_neo4j(snapshot: LibrarySnapshot, args: argparse.Namespace) -> dict:
    """Helper to load a snapshot into Neo4j."""
    require_neo4j_password(args)
    neo4j_config = build_neo4j_config(args)
    loader = Neo4jGraphLoader(neo4j_config)
    try:
        return loader.load_snapshot(snapshot)
    finally:
        loader.close()


def embed_graph(graph_id: str, args: argparse.Namespace) -> dict:
    """Helper to generate embeddings for an existing graph in Neo4j."""
    require_neo4j_password(args)
    neo4j_config = build_neo4j_config(args)
    embedding_config = build_embedding_config(args, query_mode=False)
    indexer = Neo4jEmbeddingIndexer(neo4j_config, embedding_config)
    try:
        return indexer.index_graph(graph_id)
    finally:
        indexer.close()


def build_neo4j_config(args: argparse.Namespace) -> Neo4jConfig:
    """Construct a Neo4jConfig from CLI arguments."""
    return Neo4jConfig(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
        create_vector_indexes=True,
        vector_dimensions=getattr(args, "dimensions", 256),
        vector_similarity=getattr(args, "similarity", "cosine"),
    )


def build_embedding_config(args: argparse.Namespace, *, query_mode: bool) -> EmbeddingConfig:
    """Construct an EmbeddingConfig from CLI arguments."""
    model = resolve_embedding_model(args.provider, args.model)
    batch_size = getattr(args, "batch_size", 32)
    if not query_mode and batch_size == 32 and args.provider == "gemini":
        batch_size = 8
    if not query_mode and batch_size == 32 and args.provider == "ollama":
        batch_size = 4

    batch_delay_seconds = getattr(args, "batch_delay_seconds", None)
    if batch_delay_seconds is None:
        batch_delay_seconds = 1.0 if args.provider == "gemini" and not query_mode else 0.0

    warn_unknown_model(args.provider, model)
    return EmbeddingConfig(
        provider=args.provider,
        model=model,
        dimensions=args.dimensions,
        batch_size=batch_size,
        batch_delay_seconds=batch_delay_seconds,
        max_text_length=getattr(args, "max_text_length", 12000),
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        gemini_api_key=args.gemini_api_key,
        gemini_base_url=args.gemini_base_url,
        gemini_task_type=args.gemini_task_type,
        ollama_base_url=args.ollama_base_url,
        ollama_truncate=getattr(args, "ollama_truncate", True),
        similarity=args.similarity,
        include_modules=not getattr(args, "skip_modules", False),
        include_classes=not getattr(args, "skip_classes", False),
        include_functions=not getattr(args, "skip_functions", False),
        include_examples=not getattr(args, "skip_examples", False),
        embedding_schema_version=getattr(args, "embedding_schema_version", EMBEDDING_SCHEMA_VERSION),
    )


def resolve_embedding_model(provider: str, model: str) -> str:
    """Resolve default model names for different providers."""
    if model != "hash-embedding-v1":
        return model
    if provider == "gemini":
        return "gemini-embedding-001"
    if provider == "openai":
        return "text-embedding-3-large"
    if provider == "ollama":
        return "embeddinggemma"
    return model


def warn_unknown_model(provider: str, model: str) -> None:
    """Print a warning if an embedding model is not in the known list."""
    if provider == "gemini" and model not in KNOWN_GEMINI_MODELS and not model.startswith("models/"):
        print(f"Warning: unknown Gemini embedding model; continuing with {model}.")
    if provider == "ollama" and model not in KNOWN_OLLAMA_MODELS and "/" not in model and ":" not in model:
        print(f"Warning: unknown Ollama embedding model; continuing with {model}.")


def require_neo4j_password(args: argparse.Namespace) -> None:
    """Ensure a Neo4j password is provided or exit."""
    if not args.password:
        raise SystemExit("Neo4j password is required. Pass --password or set NEO4J_PASSWORD.")


def write_or_print(payload: str, output: str | None) -> None:
    """Write string to a file or print it to stdout."""
    if output:
        Path(output).write_text(payload, encoding="utf-8")
        return
    print(payload)


def print_load_result(result: dict, database: str) -> None:
    """Format and print the results of a snapshot load."""
    print(f"Loaded graph_id={result['graph_id']} into {database}")
    print("Counts:", ", ".join(f"{key}={value}" for key, value in result.items() if key != "graph_id"))


import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_available_graphs(uri: str, username: str, password: str | None, database: str) -> list[str]:
    """Fetch a list of all library graph IDs currently in Neo4j."""
    if not password:
        return []
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(username, password), notifications_min_severity="OFF")
        with driver.session(database=database) as session:
            result = session.run("MATCH (n:Library) RETURN n.graph_id AS graph_id ORDER BY n.graph_id")
            return [row["graph_id"] for row in result]
    except Exception:
        return []
    finally:
        if 'driver' in locals():
            driver.close()

def interactive_main() -> None:
    """Entry point for the interactive configuration and workflow TUI."""
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from dotenv import load_dotenv
    from .config_ui import print_config_status, setup_neo4j_interactive, setup_embedding_interactive

    console = Console()

    while True:
        load_dotenv()
        console.print(Panel.fit("[bold cyan]Welcome to Graph RAG Interactive CLI[/bold cyan]"))
        print_config_status()

        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Ingest a local repository",
                "Search (Retrieve) from Graph Hub",
                "Agentic Chat (Decomposition + Synthesis)",
                "Configure Neo4j Database",
                "Configure Embedding Provider",
                "Exit"
            ]
        ).ask()

        if action == "Exit" or action is None:
            break

        if action == "Configure Neo4j Database":
            setup_neo4j_interactive()
            console.print("\n[cyan]Press Enter to return to main menu...[/cyan]")
            input()
            continue

        if action == "Configure Embedding Provider":
            setup_embedding_interactive()
            console.print("\n[cyan]Press Enter to return to main menu...[/cyan]")
            input()
            continue

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        if not password:
            password = questionary.password("Neo4j password (NEO4J_PASSWORD not set):").ask()
            if not password:
                continue

        # Mock an argparse namespace to reuse existing functions
        args = argparse.Namespace(
            uri=uri,
            username=username,
            password=password,
            database=database,
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

        if action == "Ingest a local repository":
            from .config_ui import select_installed_package_interactive
            path = select_installed_package_interactive()
            if not path:
                continue
            args.path = path

            args.snapshot_output = None
            args.skip_embed = False
            args.batch_size = 32
            args.batch_delay_seconds = None
            args.max_text_length = 12000
            args.skip_modules = False
            args.skip_classes = False
            args.skip_functions = False
            args.skip_examples = False
            args.embedding_schema_version = EMBEDDING_SCHEMA_VERSION

            console.print("[cyan]Ingesting repository (this may take a while)...[/cyan]")
            try:
                run_ingest(args)
                console.print("[bold green]Ingestion complete![/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error during ingestion: {e}[/bold red]")

        elif action == "Search (Retrieve) from Graph Hub":
            graphs = get_available_graphs(uri, username, password, database)
            choices = ["All"] + graphs if graphs else ["All"]
            
            graph_id = questionary.select(
                "Which library to search?",
                choices=choices
            ).ask()
            if not graph_id:
                continue
            args.graph_id = None if graph_id == "All" else graph_id

            query = questionary.text("Search query:").ask()
            if not query:
                continue
            args.query = query

            args.gemini_task_type = os.getenv("GEMINI_QUERY_TASK_TYPE", "RETRIEVAL_QUERY")
            args.top_k = 10
            args.max_entities = 10
            args.keyword_k = 8
            args.vector_k = 8
            args.hops = 2
            args.route_top_k = 3
            args.context_max_chars = 12000
            args.class_boost = 1.5
            args.public_boost = 1.25
            args.private_penalty = 0.45

            console.print("[cyan]Retrieving context...[/cyan]")
            
            neo4j_config = build_neo4j_config(args)
            embedding_config = build_embedding_config(args, query_mode=True)
            retrieval_config = RetrievalConfig(
                graph_id=args.graph_id,
                top_k=args.top_k,
                keyword_k=args.keyword_k,
                vector_k=args.vector_k,
                hops=args.hops,
                route_top_k=args.route_top_k,
                context_max_chars=args.context_max_chars,
                max_entities=args.max_entities,
                class_boost=args.class_boost,
                public_boost=args.public_boost,
                private_penalty=args.private_penalty,
            )
            retriever = Neo4jGraphRetriever(neo4j_config, embedding_config)
            try:
                result = retriever.retrieve(args.query, retrieval_config)
                lines = [f"**Graph:** `{result.graph_id}`"]
                if result.routed_graphs:
                    lines.append("**Routed graphs:** " + ", ".join(f"`{g}`" for g in result.routed_graphs))
                lines.extend(["", "### Context", "", result.compressed_context])
                md = Markdown("\n".join(lines))
                console.print(Panel(md, title="Retrieval Results", border_style="green"))
            except Exception as e:
                console.print(f"[bold red]Error during retrieval: {e}[/bold red]")
            finally:
                retriever.close()

        elif action == "Agentic Chat (Decomposition + Synthesis)":
            query = questionary.text("Enter your complex technical query:").ask()
            if not query:
                continue
            
            console.print("[cyan]Running Agentic Workflow (this may involve several LLM calls)...[/cyan]")
            from .agent_workflow import build_agent_graph
            try:
                app = build_agent_graph()
                result = app.invoke({
                    "query": query,
                    "sub_query_results": [],
                    "final_answer": ""
                })
                console.print("\n" + "="*40)
                console.print("[bold green]FINAL AGENT ANSWER[/bold green]")
                console.print("="*40)
                console.print(result.get("final_answer", "No answer generated."))
            except Exception as e:
                console.print(f"[bold red]Error during agent execution: {e}[/bold red]")

        console.print("\n[cyan]Press Enter to return to main menu...[/cyan]")
        input()



def main() -> None:
    """
    Main CLI entry point.

    Routes execution based on provided command-line arguments or
    starts the interactive TUI if no arguments are given.
    """
    if len(sys.argv) == 1:
        try:
            interactive_main()
            return
        except ImportError as e:
            print(f"Interactive mode requires additional dependencies: {e}")
            print("Please install them with: pip install -e '.[cli]'")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
