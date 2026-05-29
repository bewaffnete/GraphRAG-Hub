from pathlib import Path

from graphrag_mcp.application.dto.ingest import IngestLibraryRequest
from graphrag_mcp.application.dto.retrieve_candidates import RetrieveCandidatesRequest
from graphrag_mcp.application.use_cases.ingest_library import IngestLibraryUseCase
from graphrag_mcp.application.use_cases.retrieve_candidates import RetrieveCandidatesUseCase
from graphrag_mcp.bootstrap.container import InMemoryGraphRepository
from graphrag_mcp.domain.services.context_compressor import ContextCompressor
from graphrag_mcp.domain.services.graph_identity_policy import GraphIdentityPolicy
from graphrag_mcp.infrastructure.embeddings.hash_provider import HashEmbeddingProvider
from graphrag_mcp.infrastructure.parsing.python_ast_parser import PythonAstParser
from graphrag_mcp.infrastructure.registry.yaml_registry import InMemoryGraphRegistry


def test_retrieve_candidates_works_without_embeddings() -> None:
    repository = _build_graph(embedding_provider=None)
    use_case = RetrieveCandidatesUseCase(
        repository=repository,
        context_compressor=ContextCompressor(),
        embedding_provider=None,
    )

    response = use_case.execute(
        RetrieveCandidatesRequest(
            query="public greeting",
            graph_id="public-api-pkg:1.0.0",
            top_k=3,
        )
    )

    assert response.candidate_count >= 1
    assert response.top_matches[0].name == "public_helper"
    assert response.warnings == ["Vector search is disabled because no embedding provider is configured."]


def test_retrieve_candidates_uses_optional_vector_signal() -> None:
    embedding_provider = HashEmbeddingProvider()
    repository = _build_graph(embedding_provider=embedding_provider)
    use_case = RetrieveCandidatesUseCase(
        repository=repository,
        context_compressor=ContextCompressor(),
        embedding_provider=embedding_provider,
    )

    response = use_case.execute(
        RetrieveCandidatesRequest(
            query="public api client",
            graph_id="public-api-pkg:1.0.0",
            top_k=3,
        )
    )

    assert response.candidate_count >= 1
    assert response.top_matches[0].name == "PublicClient"
    assert response.warnings == []


def _build_graph(embedding_provider):
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "public_api_pkg"
    repository = InMemoryGraphRepository()
    ingest_use_case = IngestLibraryUseCase(
        parser=PythonAstParser(identity_policy=GraphIdentityPolicy()),
        repository=repository,
        registry=InMemoryGraphRegistry(),
        identity_policy=GraphIdentityPolicy(),
        embedding_provider=embedding_provider,
    )
    ingest_use_case.execute(
        IngestLibraryRequest(
            path=str(fixture_root),
            library_name="public_api_pkg",
            version="1.0.0",
            ingest_mode="parse_load_embed" if embedding_provider is not None else "parse_load",
            embedding_mode="enabled" if embedding_provider is not None else "disabled",
        )
    )
    return repository
