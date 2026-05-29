from graphrag_mcp.bootstrap.container import build_container


def test_container_builds_without_side_effects(monkeypatch) -> None:
    monkeypatch.setenv("GRAPHRAG_NEO4J_BACKEND", "memory")
    container = build_container()
    assert container.settings.app.app_name == "graphrag-mcp"
