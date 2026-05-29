"""Application bootstrap."""

from dataclasses import dataclass
import math
import re

from graphrag_mcp.application.use_cases.retrieve_candidates import RetrieveCandidatesUseCase
from graphrag_mcp.application.use_cases.retrieve_details import RetrieveDetailsUseCase
from graphrag_mcp.application.use_cases.list_graphs import ListGraphsUseCase
from graphrag_mcp.application.use_cases.rebuild_embeddings import RebuildEmbeddingsUseCase
from graphrag_mcp.application.use_cases.health_check import HealthCheckUseCase
from graphrag_mcp.application.use_cases.ingest_library import IngestLibraryUseCase
from graphrag_mcp.domain.services.context_compressor import ContextCompressor
from graphrag_mcp.domain.services.graph_identity_policy import GraphIdentityPolicy
from graphrag_mcp.infrastructure.config.loader import SettingsBundle, load_settings_from_env
from graphrag_mcp.infrastructure.embeddings.factory import create_embedding_provider
from graphrag_mcp.infrastructure.graph.neo4j_graph_repository import Neo4jGraphRepository
from graphrag_mcp.infrastructure.graph.neo4j_schema_manager import Neo4jSchemaManager
from graphrag_mcp.infrastructure.health_probe import DefaultHealthProbe
from graphrag_mcp.infrastructure.library_metadata import FilesystemLibraryMetadataResolver
from graphrag_mcp.infrastructure.parsing.python_ast_parser import PythonAstParser
from graphrag_mcp.infrastructure.registry.yaml_registry import InMemoryGraphRegistry


@dataclass(slots=True)
class ApplicationContainer:
    settings: SettingsBundle
    ingest_library: IngestLibraryUseCase
    list_graphs: ListGraphsUseCase
    retrieve_candidates: RetrieveCandidatesUseCase
    retrieve_details: RetrieveDetailsUseCase
    rebuild_embeddings: RebuildEmbeddingsUseCase
    health_check: HealthCheckUseCase


class InMemoryGraphRepository:
    def __init__(self) -> None:
        self.graphs = {}

    def store_library_graph(self, parsed_graph) -> None:
        self.graphs[(parsed_graph.library_name, parsed_graph.version)] = parsed_graph

    def list_graphs(self, *, name_prefix: str | None, status: str | None, limit: int) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for parsed_graph in self.graphs.values():
            library = next((node for node in parsed_graph.nodes if node.kind == "Library"), None)
            if library is None:
                continue
            name = str(library.name)
            graph_status = str(library.metadata.get("status", "active"))
            if name_prefix and not name.lower().startswith(name_prefix.lower()):
                continue
            if status and graph_status != status:
                continue
            items.append(
                {
                    "graph_id": library.graph_id,
                    "name": name,
                    "version": str(library.metadata.get("version", parsed_graph.version)),
                    "status": graph_status,
                    "node_counts": _count_nodes(parsed_graph.nodes),
                    "embedding_status": _embedding_status_from_nodes(parsed_graph.nodes),
                    "updated_at": library.updated_at,
                }
            )
        return items[:limit]

    def get_nodes_by_ids(self, node_ids: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
        requested = set(node_ids)
        found_nodes: list[dict[str, object]] = []
        found_ids: set[str] = set()
        all_edges: list[dict[str, object]] = []
        for parsed_graph in self.graphs.values():
            for node in parsed_graph.nodes:
                if node.node_id in requested:
                    found_nodes.append(
                        {
                            "node_id": node.node_id,
                            "graph_id": node.graph_id,
                            "kind": node.kind,
                            "name": node.name,
                            "qualified_name": node.qualified_name,
                            "signature": node.signature,
                            "docstring": node.docstring,
                            "logic_skeleton": node.logic_skeleton,
                            "source_excerpt": node.source_excerpt,
                            "metadata": {"source_path": node.source_path, "line_start": node.line_start, "line_end": node.line_end, **node.metadata},
                        }
                    )
                    found_ids.add(node.node_id)
            for edge in parsed_graph.edges:
                if edge.source_node_id in requested and edge.target_node_id in requested:
                    all_edges.append(
                        {
                            "source_node_id": edge.source_node_id,
                            "target_node_id": edge.target_node_id,
                            "type": edge.type,
                        }
                    )
        missing = [node_id for node_id in node_ids if node_id not in found_ids]
        return found_nodes, all_edges, missing

    def rebuild_embeddings(self, *, graph_id: str, embedding_provider) -> tuple[int, int]:
        for parsed_graph in self.graphs.values():
            if any(node.graph_id == graph_id for node in parsed_graph.nodes):
                before = sum(1 for node in parsed_graph.nodes if node.kind in {"Module", "Class", "Function", "Example"} and "embedding" in node.metadata)
                embedded = embedding_provider.embed_graph(parsed_graph)
                skipped = max(0, before - embedded)
                return embedded, skipped
        return 0, 0

    def search_candidates(self, *, query: str, graph_ids: list[str] | None, labels: list[str] | None, top_k: int, query_embedding: list[float] | None):
        from graphrag_mcp.domain.services.scoring_policy import ScoringPolicy
        from graphrag_mcp.domain.value_objects.candidate_match import CandidateMatch

        scoring = ScoringPolicy()
        terms = _tokenize_text(query)
        corpus = []
        matches = []
        for parsed_graph in self.graphs.values():
            for node in parsed_graph.nodes:
                if node.kind == "Library":
                    continue
                if graph_ids and node.graph_id not in graph_ids:
                    continue
                if labels and node.kind not in labels:
                    continue
                haystack = " ".join(
                    [
                        node.name,
                        node.qualified_name,
                        node.summary or "",
                        node.docstring or "",
                        node.signature or "",
                        str(node.metadata.get("embedding_text", "")),
                    ]
                ).lower()
                corpus.append((node, _tokenize_text(haystack), haystack))
        bm25_scores = _bm25_scores(terms, corpus)
        for node, _, haystack in corpus:
                keyword_hits = sum(1 for term in terms if term in haystack)
                exact_match = query.lower() in {node.name.lower(), node.qualified_name.lower()}
                vector_score = 0.0
                if query_embedding and isinstance(node.metadata.get("embedding"), list):
                    vector_score = _cosine_similarity(query_embedding, node.metadata["embedding"])
                bm25_score = bm25_scores.get(node.node_id, 0.0)
                if keyword_hits == 0 and bm25_score == 0.0 and vector_score == 0.0 and not exact_match:
                    continue
                keyword_score = bm25_score if bm25_score > 0.0 else keyword_hits / max(len(terms), 1)
                score = scoring.score(
                    keyword_score=keyword_score,
                    vector_score=vector_score,
                    is_public=node.is_public,
                    kind=node.kind,
                    exact_match=exact_match,
                )
                matches.append(
                    CandidateMatch(
                        node_id=node.node_id,
                        graph_id=node.graph_id,
                        kind=node.kind,
                        name=node.name,
                        qualified_name=node.qualified_name,
                        summary=node.summary or "",
                        score=score,
                        source_path=node.source_path,
                        keyword_score=keyword_score,
                        vector_score=vector_score,
                    )
                )
        matches.sort(key=lambda item: (item.score, item.keyword_score, item.vector_score, item.qualified_name), reverse=True)
        return matches[:top_k]


def build_container() -> ApplicationContainer:
    settings = load_settings_from_env()
    identity_policy = GraphIdentityPolicy()
    repository = _build_repository(settings)
    registry = InMemoryGraphRegistry()
    embedding_provider = create_embedding_provider(settings.embedding) if settings.embedding.enabled else None
    ingest_library = IngestLibraryUseCase(
        parser=PythonAstParser(identity_policy=identity_policy),
        repository=repository,
        registry=registry,
        identity_policy=identity_policy,
        embedding_provider=embedding_provider,
        metadata_resolver=FilesystemLibraryMetadataResolver(),
    )
    retrieve_candidates = RetrieveCandidatesUseCase(
        repository=repository,
        context_compressor=ContextCompressor(),
        embedding_provider=embedding_provider,
    )
    list_graphs = ListGraphsUseCase(repository=repository)
    retrieve_details = RetrieveDetailsUseCase(repository=repository, context_compressor=ContextCompressor())
    rebuild_embeddings = RebuildEmbeddingsUseCase(repository=repository, embedding_provider=embedding_provider)
    health_check = HealthCheckUseCase(
        health_probe=DefaultHealthProbe(settings=settings, repository=repository, registry=registry),
        server_version=settings.app.version,
    )
    return ApplicationContainer(
        settings=settings,
        ingest_library=ingest_library,
        list_graphs=list_graphs,
        retrieve_candidates=retrieve_candidates,
        retrieve_details=retrieve_details,
        rebuild_embeddings=rebuild_embeddings,
        health_check=health_check,
    )


def _build_repository(settings: SettingsBundle):
    if settings.neo4j.backend == "neo4j":
        return Neo4jGraphRepository(settings.neo4j, schema_manager=Neo4jSchemaManager())
    return InMemoryGraphRepository()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return round(sum(a * b for a, b in zip(left, right, strict=False)), 6)


def _tokenize_text(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if token]


def _bm25_scores(terms: list[str], corpus: list[tuple[object, list[str], str]]) -> dict[str, float]:
    if not terms or not corpus:
        return {}
    doc_count = len(corpus)
    avgdl = sum(len(tokens) for _, tokens, _ in corpus) / doc_count
    k1 = 1.5
    b = 0.75
    doc_freq: dict[str, int] = {}
    for _, tokens, _ in corpus:
        seen = set(tokens)
        for term in terms:
            if term in seen:
                doc_freq[term] = doc_freq.get(term, 0) + 1
    scores: dict[str, float] = {}
    for node, tokens, _ in corpus:
        term_freqs: dict[str, int] = {}
        for token in tokens:
            if token in terms:
                term_freqs[token] = term_freqs.get(token, 0) + 1
        score = 0.0
        doc_len = len(tokens) or 1
        for term in terms:
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
            denom = tf + k1 * (1 - b + b * (doc_len / avgdl if avgdl else 1.0))
            score += idf * ((tf * (k1 + 1)) / denom)
        if score > 0:
            scores[node.node_id] = round(score, 6)
    return scores


def _count_nodes(nodes) -> dict[str, int]:
    counts = {"modules": 0, "classes": 0, "functions": 0, "examples": 0}
    for node in nodes:
        if node.kind == "Module":
            counts["modules"] += 1
        elif node.kind == "Class":
            counts["classes"] += 1
        elif node.kind == "Function":
            counts["functions"] += 1
        elif node.kind == "Example":
            counts["examples"] += 1
    return counts


def _embedding_status_from_nodes(nodes) -> dict[str, object]:
    embedded = [node for node in nodes if "embedding" in node.metadata]
    first = embedded[0] if embedded else None
    return {
        "ready": bool(embedded),
        "provider": first.metadata.get("embedding_provider") if first else None,
        "model": first.metadata.get("embedding_model") if first else None,
        "schema_version": first.metadata.get("embedding_schema_version") if first else None,
    }
