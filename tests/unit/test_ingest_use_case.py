from pathlib import Path

from graphrag_mcp.application.dto.ingest import IngestLibraryRequest
from graphrag_mcp.application.use_cases.ingest_library import IngestLibraryUseCase
from graphrag_mcp.domain.services.graph_identity_policy import GraphIdentityPolicy
from graphrag_mcp.infrastructure.embeddings.hash_provider import HashEmbeddingProvider
from graphrag_mcp.infrastructure.parsing.python_ast_parser import PythonAstParser
from graphrag_mcp.infrastructure.registry.yaml_registry import InMemoryGraphRegistry


class FakeGraphRepository:
    def __init__(self) -> None:
        self.saved = None

    def store_library_graph(self, parsed_graph) -> None:
        self.saved = parsed_graph


def test_ingest_use_case_runs_public_api_pipeline() -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "public_api_pkg"
    repository = FakeGraphRepository()
    use_case = IngestLibraryUseCase(
        parser=PythonAstParser(identity_policy=GraphIdentityPolicy()),
        repository=repository,
        registry=InMemoryGraphRegistry(),
        identity_policy=GraphIdentityPolicy(),
        embedding_provider=HashEmbeddingProvider(),
    )

    response = use_case.execute(
        IngestLibraryRequest(
            path=str(fixture_root),
            library_name="public_api_pkg",
            version="1.0.0",
            ingest_mode="parse_load_embed",
            embedding_mode="enabled",
        )
    )

    assert response.graph_id == "public-api-pkg:1.0.0"
    assert "filter_to_public_api" in response.executed_stages
    assert response.counts["classes"] == 1
    assert response.counts["functions"] >= 2
    assert repository.saved is not None
    embedded_nodes = [node for node in repository.saved.nodes if node.metadata.get("embedding")]
    assert embedded_nodes


def test_ingest_use_case_resolves_package_name_from_user_venv(monkeypatch, tmp_path) -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "public_api_pkg"
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    package_link = site_packages / "public_api_pkg"
    package_link.symlink_to(fixture_root, target_is_directory=True)
    monkeypatch.setenv("GRAPHRAG_VENV_LIBS_CONTAINER_PATH", str(site_packages))

    repository = FakeGraphRepository()
    use_case = IngestLibraryUseCase(
        parser=PythonAstParser(identity_policy=GraphIdentityPolicy()),
        repository=repository,
        registry=InMemoryGraphRegistry(),
        identity_policy=GraphIdentityPolicy(),
        embedding_provider=HashEmbeddingProvider(),
    )

    response = use_case.execute(
        IngestLibraryRequest(
            path="public-api-pkg",
            library_name="public_api_pkg",
            version="1.0.0",
            ingest_mode="parse_load",
            embedding_mode="disabled",
        )
    )

    assert response.graph_id == "public-api-pkg:1.0.0"
    assert repository.saved is not None


def test_ingest_use_case_resolves_library_name_when_path_is_site_packages_root(monkeypatch, tmp_path) -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "public_api_pkg"
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / "aiohappyeyeballs").mkdir()
    package_link = site_packages / "public_api_pkg"
    package_link.symlink_to(fixture_root, target_is_directory=True)
    monkeypatch.setenv("GRAPHRAG_VENV_LIBS_CONTAINER_PATH", str(site_packages))

    repository = FakeGraphRepository()
    use_case = IngestLibraryUseCase(
        parser=PythonAstParser(identity_policy=GraphIdentityPolicy()),
        repository=repository,
        registry=InMemoryGraphRegistry(),
        identity_policy=GraphIdentityPolicy(),
        embedding_provider=HashEmbeddingProvider(),
    )

    response = use_case.execute(
        IngestLibraryRequest(
            path=str(site_packages),
            library_name="public_api_pkg",
            version="1.0.0",
            ingest_mode="parse_load",
            embedding_mode="disabled",
        )
    )

    assert response.graph_id == "public-api-pkg:1.0.0"
    assert repository.saved is not None
    assert repository.saved.root_path.resolve() == fixture_root.resolve()
