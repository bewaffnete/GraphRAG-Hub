from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import time
from typing import Any, Protocol
from urllib import request

from .neo4j_loader import Neo4jConfig

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - handled at runtime with a clear error
    GraphDatabase = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

EMBEDDING_SCHEMA_VERSION = "2026-05-13-public-api-v2"


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding generation and indexing.

    Attributes:
        provider (str): Embedding provider ('hash', 'openai', 'gemini', 'ollama').
        model (str): Specific model name to use.
        dimensions (int): Vector dimensionality.
        batch_size (int): Number of texts to embed in one request.
        batch_delay_seconds (float): Delay between batches to avoid rate limits.
        max_text_length (int): Maximum length of text to embed before truncation.
        openai_api_key (str | None): API key for OpenAI.
        openai_base_url (str): Base URL for OpenAI API.
        gemini_api_key (str | None): API key for Gemini.
        gemini_base_url (str): Base URL for Gemini API.
        gemini_task_type (str): Task type for Gemini embeddings.
        ollama_base_url (str): Base URL for Ollama API.
        ollama_truncate (bool): Whether Ollama should truncate long inputs.
        similarity (str): Similarity function ('cosine', 'euclidean').
        include_modules (bool): Whether to index modules.
        include_classes (bool): Whether to index classes.
        include_functions (bool): Whether to index functions.
        include_examples (bool): Whether to index code examples.
        embedding_schema_version (str): Internal versioning for embeddings.
    """
    provider: str = "hash"
    model: str = "hash-embedding-v1"
    dimensions: int = 256
    batch_size: int = 32
    batch_delay_seconds: float = 0.0
    max_text_length: int = 12000
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_task_type: str = "RETRIEVAL_DOCUMENT"
    ollama_base_url: str = "http://localhost:11434"
    ollama_truncate: bool = True
    similarity: str = "cosine"
    include_modules: bool = True
    include_classes: bool = True
    include_functions: bool = True
    include_examples: bool = True
    embedding_schema_version: str = EMBEDDING_SCHEMA_VERSION


@dataclass
class EmbeddingDocument:
    """Represents a piece of text ready to be embedded."""
    graph_id: str
    label: str
    embedding_text: str
    properties: dict[str, Any]


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...


class HashEmbeddingProvider:
    """A deterministic embedding provider based on SHA-256 hashing."""
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(text, self.dimensions) for text in texts]


class OpenAIEmbeddingProvider:
    """Embedding provider using OpenAI's API."""
    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/embeddings",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with request.urlopen(req) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        ordered = sorted(data["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


class GeminiEmbeddingProvider:
    """Embedding provider using Google's Gemini API."""
    def __init__(self, api_key: str, model: str, base_url: str, output_dimensionality: int, task_type: str):
        self.api_key = api_key
        self.model = model.removeprefix("models/")
        self.base_url = base_url.rstrip("/")
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type

    def embed(self, texts: list[str]) -> list[list[float]]:
        requests_payload = []
        for text in texts:
            requests_payload.append(
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": self.task_type,
                    "outputDimensionality": self.output_dimensionality,
                }
            )

        payload = json.dumps({"requests": requests_payload}).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/models/{self.model}:batchEmbedContents",
            data=payload,
            method="POST",
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        with request.urlopen(req) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))

        responses = data.get("embeddings")
        if responses is None:
            responses = [item.get("embedding") for item in data.get("responses", [])]
        if not responses:
            raise RuntimeError("Gemini API returned no embeddings.")
        return [item["values"] for item in responses]


class OllamaEmbeddingProvider:
    """Embedding provider using a local Ollama instance."""
    def __init__(self, model: str, base_url: str, dimensions: int | None, truncate: bool):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimensions = dimensions
        self.truncate = truncate

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "truncate": self.truncate,
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        req = request.Request(
            url=f"{self.base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}. Ensure Ollama is running and OLLAMA_BASE_URL is correct. Error: {e}") from e

        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError("Ollama API returned no embeddings.")
        return embeddings


class Neo4jEmbeddingIndexer:
    """
    Indexer that computes embeddings for graph nodes and stores them in Neo4j.

    It handles schema creation (vector and fulltext indexes), stale data cleanup,
    and batch processing of nodes.
    """

    def __init__(self, neo4j_config: Neo4jConfig, embedding_config: EmbeddingConfig):
        """
        Initialize the indexer.

        Args:
            neo4j_config (Neo4jConfig): Neo4j connection settings.
            embedding_config (EmbeddingConfig): Embedding provider settings.
        """
        if GraphDatabase is None:
            raise RuntimeError(
                "The 'neo4j' package is not installed. Activate the target environment and install it first."
            )
        self.neo4j_config = neo4j_config
        self.embedding_config = embedding_config
        self.driver = GraphDatabase.driver(
            neo4j_config.uri,
            auth=(neo4j_config.username, neo4j_config.password),
            notifications_min_severity="OFF"
        )
        self.provider = build_embedding_provider(embedding_config)

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()

    def index_graph(self, graph_id: str) -> dict[str, int | str]:
        """
        Generate and store embeddings for all applicable nodes in a graph.

        Args:
            graph_id (str): The unique ID of the graph to index.

        Returns:
            dict: Statistics about the indexing process.
        """
        with self.driver.session(database=self.neo4j_config.database) as session:
            session.execute_write(self._ensure_fulltext_indexes)
            session.execute_write(self._clear_stale_embeddings, graph_id, self.embedding_config.embedding_schema_version)
            documents = session.execute_read(self._collect_documents, graph_id, self.embedding_config)

            embedded = 0
            vector_indexes_ready = False
            progress = _build_progress_bar(len(documents), graph_id, self.embedding_config.model)
            try:
                for offset in range(0, len(documents), self.embedding_config.batch_size):
                    batch = documents[offset : offset + self.embedding_config.batch_size]
                    if progress is not None:
                        progress.set_postfix(batch=len(batch), refresh=False)
                    vectors = self.provider.embed([doc.embedding_text for doc in batch])
                    if vectors and not vector_indexes_ready:
                        session.execute_write(self._ensure_vector_indexes, len(vectors[0]))
                        vector_indexes_ready = True
                    rows = [
                        {
                            "graph_id": doc.graph_id,
                            "label": doc.label,
                            "embedding_text": doc.embedding_text,
                            "embedding_model": self.embedding_config.model,
                            "embedding_dimensions": len(vector),
                            "embedding_schema_version": self.embedding_config.embedding_schema_version,
                            "embedded_at": _utc_now(),
                            "embedding": vector,
                            **doc.properties,
                        }
                        for doc, vector in zip(batch, vectors)
                    ]
                    session.execute_write(self._write_embeddings, rows)
                    embedded += len(rows)
                    if progress is not None:
                        progress.update(len(rows))
                    if self.embedding_config.batch_delay_seconds > 0 and offset + self.embedding_config.batch_size < len(documents):
                        time.sleep(self.embedding_config.batch_delay_seconds)
            finally:
                if progress is not None:
                    progress.close()

        return {"graph_id": graph_id, "embedded_nodes": embedded}

    @staticmethod
    def _clear_stale_embeddings(tx, graph_id: str, embedding_schema_version: str) -> None:
        """Remove embeddings that use an outdated schema version."""
        tx.run(
            """
            MATCH (n)
            WHERE n.graph_id STARTS WITH $prefix
              AND n.embedding IS NOT NULL
              AND coalesce(n.embedding_schema_version, '') <> $embedding_schema_version
            REMOVE n.embedding,
                   n.embedding_text,
                   n.embedding_model,
                   n.embedding_dimensions,
                   n.embedding_updated_at,
                   n.embedding_schema_version
            """,
            prefix=f"{graph_id}:",
            embedding_schema_version=embedding_schema_version,
        )

    @staticmethod
    def _ensure_fulltext_indexes(tx) -> None:
        """Create fulltext search indexes if they don't exist."""
        statements = [
            "CREATE FULLTEXT INDEX module_content_fulltext IF NOT EXISTS FOR (n:Module) ON EACH [n.name, n.file_path, n.docstring, n.embedding_text]",
            "CREATE FULLTEXT INDEX class_content_fulltext IF NOT EXISTS FOR (n:Class) ON EACH [n.name, n.qualname, n.docstring, n.embedding_text]",
            "CREATE FULLTEXT INDEX function_content_fulltext IF NOT EXISTS FOR (n:Function) ON EACH [n.name, n.qualname, n.signature, n.docstring, n.embedding_text]",
            "CREATE FULLTEXT INDEX example_content_fulltext IF NOT EXISTS FOR (n:Example) ON EACH [n.code, n.description, n.embedding_text]",
        ]
        for statement in statements:
            tx.run(statement)

    def _ensure_vector_indexes(self, tx, dimensions: int) -> None:
        """Create vector similarity indexes for supported node types."""
        statements = []
        similarity = self.embedding_config.similarity
        if self.embedding_config.include_modules:
            statements.append(_vector_index_statement("module_embedding_vector", "Module", dimensions, similarity))
        if self.embedding_config.include_classes:
            statements.append(_vector_index_statement("class_embedding_vector", "Class", dimensions, similarity))
        if self.embedding_config.include_functions:
            statements.append(_vector_index_statement("function_embedding_vector", "Function", dimensions, similarity))
        if self.embedding_config.include_examples:
            statements.append(_vector_index_statement("example_embedding_vector", "Example", dimensions, similarity))
        for statement in statements:
            tx.run(statement)

    @staticmethod
    def _collect_documents(tx, graph_id: str, config: EmbeddingConfig) -> list[EmbeddingDocument]:
        """Fetch node data from Neo4j and format it for embedding."""
        documents: list[EmbeddingDocument] = []
        if config.include_modules:
            result = tx.run(
                """
                MATCH (n:Module)
                WHERE n.graph_id STARTS WITH $prefix
                RETURN n.graph_id AS graph_id, n.name AS name, n.file_path AS file_path, n.docstring AS docstring
                ORDER BY n.graph_id
                """,
                prefix=f"{graph_id}:",
            )
            for row in result:
                text = build_module_embedding_text(row["name"], row["file_path"], row["docstring"], config.max_text_length)
                documents.append(EmbeddingDocument(row["graph_id"], "Module", text, {}))

        if config.include_classes:
            result = tx.run(
                """
                MATCH (n:Class)
                WHERE n.graph_id STARTS WITH $prefix
                RETURN n.graph_id AS graph_id,
                       n.name AS name,
                       n.qualname AS qualname,
                       n.module AS module,
                       n.bases AS bases,
                       n.docstring AS docstring,
                       n.is_public AS is_public,
                       n.api_rank AS api_rank
                ORDER BY n.graph_id
                """,
                prefix=f"{graph_id}:",
            )
            for row in result:
                text = build_class_embedding_text(
                    row["name"],
                    row["qualname"],
                    row["module"],
                    row["bases"] or [],
                    row["docstring"],
                    bool(row["is_public"]),
                    row["api_rank"] or 1.0,
                    config.max_text_length,
                )
                documents.append(EmbeddingDocument(row["graph_id"], "Class", text, {}))

        if config.include_functions:
            result = tx.run(
                """
                MATCH (n:Function)
                WHERE n.graph_id STARTS WITH $prefix
                OPTIONAL MATCH (n)-[:HAS_PARAM]->(p:Parameter)
                OPTIONAL MATCH (n)-[:RETURNS]->(t:Type)
                OPTIONAL MATCH (n)-[:RAISES]->(e:Exception)
                WITH n,
                     collect(DISTINCT p) AS params,
                     collect(DISTINCT t) AS returns,
                     collect(DISTINCT e) AS raises
                RETURN n.graph_id AS graph_id,
                       n.name AS name,
                       n.qualname AS qualname,
                       n.module AS module,
                       n.signature AS signature,
                       n.docstring AS docstring,
                       n.is_method AS is_method,
                       n.is_async AS is_async,
                       n.is_public AS is_public,
                       n.api_rank AS api_rank,
                       n.parent_class AS parent_class,
                       [param IN params WHERE param.graph_id IS NOT NULL | {
                           name: param.name,
                           type_hint: param.type_hint,
                           default: param.default,
                           description: param.description,
                           kind: param.kind,
                           position: param.position
                       }] AS parameters,
                       [ret IN returns WHERE ret.graph_id IS NOT NULL | {
                           name: ret.name,
                           description: ret.description
                       }] AS returns,
                       [err IN raises WHERE err.graph_id IS NOT NULL | {
                           name: err.name,
                           description: err.description
                       }] AS raises
                ORDER BY n.graph_id
                """,
                prefix=f"{graph_id}:",
            )
            for row in result:
                text = build_function_embedding_text(
                    name=row["name"],
                    qualname=row["qualname"],
                    module=row["module"],
                    signature=row["signature"],
                    docstring=row["docstring"],
                    parameters=row["parameters"],
                    returns=row["returns"],
                    raises=row["raises"],
                    is_method=bool(row["is_method"]),
                    is_async=bool(row["is_async"]),
                    is_public=bool(row["is_public"]),
                    api_rank=row["api_rank"] or 1.0,
                    parent_class=row["parent_class"],
                    max_length=config.max_text_length,
                )
                documents.append(EmbeddingDocument(row["graph_id"], "Function", text, {}))

        if config.include_examples:
            result = tx.run(
                """
                MATCH (n:Example)
                WHERE n.graph_id STARTS WITH $prefix
                RETURN n.graph_id AS graph_id, n.code AS code, n.description AS description, n.source AS source, n.owner_label AS owner_label
                ORDER BY n.graph_id
                """,
                prefix=f"{graph_id}:",
            )
            for row in result:
                text = build_example_embedding_text(
                    row["code"],
                    row["description"],
                    row["source"],
                    row["owner_label"],
                    config.max_text_length,
                )
                documents.append(EmbeddingDocument(row["graph_id"], "Example", text, {}))

        return documents

    @staticmethod
    def _write_embeddings(tx, rows: list[dict[str, Any]]) -> None:
        """Save computed embeddings back to Neo4j nodes."""
        for label in ("Module", "Class", "Function", "Example"):
            batch = [row for row in rows if row["label"] == label]
            if not batch:
                continue
            tx.run(
                f"""
                UNWIND $rows AS row
                MATCH (n:{label} {{graph_id: row.graph_id}})
                SET n.embedding = row.embedding,
                    n.embedding_text = row.embedding_text,
                    n.embedding_model = row.embedding_model,
                    n.embedding_dimensions = row.embedding_dimensions,
                    n.embedding_schema_version = row.embedding_schema_version,
                    n.embedding_updated_at = row.embedded_at
                """,
                rows=batch,
            )


def build_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Factory function to build an embedding provider from configuration."""
    provider = config.provider.lower()
    if provider == "hash":
        return HashEmbeddingProvider(config.dimensions)
    if provider == "openai":
        api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai.")
        return OpenAIEmbeddingProvider(api_key=api_key, model=config.model, base_url=config.openai_base_url)
    if provider == "gemini":
        api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for provider=gemini.")
        return GeminiEmbeddingProvider(
            api_key=api_key,
            model=config.model,
            base_url=config.gemini_base_url,
            output_dimensionality=config.dimensions,
            task_type=config.gemini_task_type,
        )
    if provider == "ollama":
        return OllamaEmbeddingProvider(
            model=config.model,
            base_url=config.ollama_base_url,
            dimensions=config.dimensions,
            truncate=config.ollama_truncate,
        )
    raise ValueError(f"Unsupported embedding provider: {config.provider}")


def extract_summary(docstring: str | None) -> str:
    """Extract the first paragraph of a docstring as a summary."""
    if not docstring:
        return ""
    lines = docstring.strip().splitlines()
    if not lines:
        return ""

    summary_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if summary_lines:
                break
            continue
        summary_lines.append(stripped)
    return " ".join(summary_lines)


def clean_signature(signature: str | None) -> str:
    """Remove 'self' and 'cls' from the signature and clean it up."""
    if not signature:
        return ""
    # Remove 'self, ' or 'cls, '
    cleaned = re.sub(r"\b(self|cls),\s*", "", signature)
    # Remove '(self)' or '(cls)'
    cleaned = re.sub(r"\(\s*(self|cls)\s*\)", "()", cleaned)
    return cleaned


def build_module_embedding_text(name: str, file_path: str | None, docstring: str | None, max_length: int) -> str:
    """Construct the text representation of a module for embedding."""
    parts = [f"{name}", "Type: Module"]

    # Remove internal technical paths
    if file_path and not (file_path.startswith("_") or "/_" in file_path or "utils/" in file_path):
        parts.append(f"Path: {file_path}")

    summary = extract_summary(docstring)
    if summary:
        parts.append(f"Summary: {summary}")

    return _truncate("\n".join(parts), max_length)


def build_class_embedding_text(
    name: str,
    qualname: str,
    module: str | None,
    bases: list[str],
    docstring: str | None,
    is_public: bool,
    api_rank: float,
    max_length: int,
) -> str:
    """Construct the text representation of a class for embedding."""
    parts = [
        f"{qualname}",
        "Type: Class",
    ]
    if bases:
        parts.append(f"Bases: {', '.join(bases)}")

    summary = extract_summary(docstring)
    if summary:
        parts.append(f"Summary: {summary}")

    return _truncate("\n".join(parts), max_length)


def build_function_embedding_text(
    *,
    name: str,
    qualname: str,
    module: str | None,
    signature: str | None,
    docstring: str | None,
    parameters: list[dict[str, Any]],
    returns: list[dict[str, Any]],
    raises: list[dict[str, Any]],
    is_method: bool,
    is_async: bool,
    is_public: bool,
    api_rank: float,
    parent_class: str | None,
    max_length: int,
) -> str:
    """Construct the text representation of a function/method for embedding."""
    parts = [
        f"{qualname}",
        f"Type: {'Method' if is_method else 'Function'}",
    ]

    cleaned_sig = clean_signature(signature)
    if cleaned_sig:
        parts.append(f"Signature: {cleaned_sig}")

    summary = extract_summary(docstring)
    if summary:
        parts.append(f"Summary: {summary}")

    return _truncate("\n".join(parts), max_length)


def build_example_embedding_text(
    code: str,
    description: str | None,
    source: str | None,
    owner_label: str | None,
    max_length: int,
) -> str:
    """Construct the text representation of a code example for embedding."""
    parts = []
    if owner_label:
        parts.append(f"Owner: {owner_label}")
    if source:
        parts.append(f"Source: {source}")
    if description:
        parts.append(f"Description: {description}")
    parts.append(f"Code:\n{code}")
    return _truncate("\n".join(parts), max_length)


def _hash_embed(text: str, dimensions: int) -> list[float]:
    """Deterministic hash-based embedding fallback."""
    vector = [0.0] * dimensions
    for token in text.split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = -1.0 if digest[4] % 2 else 1.0
        weight = 1.0 + (digest[5] / 255.0)
        vector[index] += sign * weight
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to a maximum length with a suffix."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 14] + "\n\n[truncated]"


def _vector_index_statement(index_name: str, label: str, dimensions: int, similarity: str) -> str:
    """Generate a Cypher statement to create a vector index."""
    return (
        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.embedding) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: '{similarity}'}}}}"
    )


def _utc_now() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _build_progress_bar(total: int, graph_id: str, model: str):
    """Optionally build a tqdm progress bar."""
    if tqdm is None:
        return None
    return tqdm(
        total=total,
        desc=f"Embedding {graph_id}",
        unit="node",
        dynamic_ncols=True,
        mininterval=0.2,
        leave=True,
    )
