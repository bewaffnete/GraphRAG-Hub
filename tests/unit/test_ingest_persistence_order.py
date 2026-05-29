from pathlib import Path
from copy import deepcopy

from graphrag_mcp.application.dto.ingest import IngestLibraryRequest
from graphrag_mcp.application.use_cases.ingest_library import IngestLibraryUseCase
from graphrag_mcp.domain.services.graph_identity_policy import GraphIdentityPolicy
from graphrag_mcp.infrastructure.embeddings.hash_provider import HashEmbeddingProvider
from graphrag_mcp.infrastructure.parsing.python_ast_parser import PythonAstParser
from graphrag_mcp.infrastructure.registry.yaml_registry import InMemoryGraphRegistry


class SnapshotRepository:
    def __init__(self) -> None:
        self.snapshot = None

    def store_library_graph(self, parsed_graph) -> None:
        self.snapshot = deepcopy(parsed_graph)


def test_embeddings_are_present_before_persist() -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "public_api_pkg"
    repository = SnapshotRepository()
    use_case = IngestLibraryUseCase(
        parser=PythonAstParser(identity_policy=GraphIdentityPolicy()),
        repository=repository,
        registry=InMemoryGraphRegistry(),
        identity_policy=GraphIdentityPolicy(),
        embedding_provider=HashEmbeddingProvider(),
    )

    use_case.execute(
        IngestLibraryRequest(
            path=str(fixture_root),
            library_name="public_api_pkg",
            version="1.0.0",
            ingest_mode="parse_load_embed",
            embedding_mode="enabled",
        )
    )

    assert repository.snapshot is not None
    embedded = [
        node for node in repository.snapshot.nodes
        if node.kind in {"Module", "Class", "Function", "Example"} and "embedding" in node.metadata
    ]
    assert embedded
