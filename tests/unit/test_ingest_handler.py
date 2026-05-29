from pathlib import Path

from graphrag_mcp.application.use_cases.ingest_library import IngestLibraryUseCase
from graphrag_mcp.domain.services.graph_identity_policy import GraphIdentityPolicy
from graphrag_mcp.infrastructure.parsing.python_ast_parser import PythonAstParser
from graphrag_mcp.infrastructure.registry.yaml_registry import InMemoryGraphRegistry
from graphrag_mcp.interfaces.mcp.handlers.ingest_library import handle_graphrag_ingest_library


class FakeGraphRepository:
    def store_library_graph(self, parsed_graph) -> None:
        self.saved = parsed_graph


def test_ingest_handler_returns_contract_shape() -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "public_api_pkg"
    use_case = IngestLibraryUseCase(
        parser=PythonAstParser(identity_policy=GraphIdentityPolicy()),
        repository=FakeGraphRepository(),
        registry=InMemoryGraphRegistry(),
        identity_policy=GraphIdentityPolicy(),
        embedding_provider=None,
    )

    response = handle_graphrag_ingest_library(
        {
            "path": str(fixture_root),
            "library_name": "public_api_pkg",
            "version": "1.0.0",
            "ingest_mode": "parse_load",
            "embedding_mode": "disabled",
        },
        use_case,
    )

    assert response["graph_id"] == "public-api-pkg:1.0.0"
    assert "counts" in response
    assert "warnings" in response
    assert "executed_stages" in response
