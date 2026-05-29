from pathlib import Path

from graphrag_mcp.domain.services.graph_identity_policy import GraphIdentityPolicy
from graphrag_mcp.infrastructure.parsing.python_ast_parser import PythonAstParser


def test_parser_keeps_only_public_api_nodes() -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "public_api_pkg"
    parser = PythonAstParser(identity_policy=GraphIdentityPolicy())
    graph = parser.parse_public_api(
        root_path=fixture_root,
        library_name="public_api_pkg",
        version="1.0.0",
        graph_id="public-api-pkg:1.0.0",
    )

    qualified_names = {node.qualified_name for node in graph.nodes}
    assert "public_api_pkg.api.PublicClient" in qualified_names
    assert "public_api_pkg.api.public_helper" in qualified_names
    assert "public_api_pkg._internal.InternalOnly" not in qualified_names
    assert "public_api_pkg.api._private_helper" not in qualified_names


def test_parser_extracts_parameters_returns_exceptions_and_examples() -> None:
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures" / "public_api_pkg"
    parser = PythonAstParser(identity_policy=GraphIdentityPolicy())
    graph = parser.parse_public_api(
        root_path=fixture_root,
        library_name="public_api_pkg",
        version="1.0.0",
        graph_id="public-api-pkg:1.0.0",
    )

    kinds = [node.kind for node in graph.nodes]
    assert "Parameter" in kinds
    assert "ReturnType" in kinds
    assert "Exception" in kinds
    assert "Example" in kinds
