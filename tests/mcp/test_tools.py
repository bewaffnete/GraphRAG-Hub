import pytest
from unittest.mock import patch, MagicMock

from graph_rag.mcp.schemas import RetrieveInput, ListGraphsInput
from graph_rag.mcp.tools import execute_retrieve, execute_list_graphs
from graph_rag.query_decomposition import register_graph_in_config, load_available_graphs

@pytest.mark.asyncio
async def test_execute_retrieve():
    input_data = RetrieveInput(query="test query", graph_id="test:1.0", top_k=5)

    with patch("graph_rag.mcp.tools.build_agent_graph") as mock_build_agent:
        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "final_answer": "Use snapshot.save_html('report.html').",
            "retrieved_contexts": ["Graph: selected_context\nRelevant implementation:\n[Class] TestClass"],
            "selected_ids": ["test:1.0:TestClass", "test:1.0:TestClass"],
            "candidates": [
                {
                    "id": "test:1.0:TestClass",
                    "name": "TestClass",
                    "type": "Class",
                    "summary": "Test docstring",
                },
                {
                    "id": "test:1.0:Noise",
                    "name": "Noise",
                    "type": "Example",
                    "summary": "Irrelevant",
                },
                {
                    "id": "test:1.0:TestClass",
                    "name": "TestClass",
                    "type": "Class",
                    "summary": "",
                },
            ],
        }
        mock_build_agent.return_value = mock_app

        result = await execute_retrieve(input_data)

        mock_app.invoke.assert_called_once_with(
            {
                "query": "test query",
                "graph_id": "test:1.0",
                "top_k": 5,
                "candidates": [],
                "selected_ids": [],
                "retrieved_contexts": [],
                "final_answer": "",
            }
        )

        assert len(result) == 1
        assert result[0]["query"] == "test query"
        assert result[0]["answer"] == "Use snapshot.save_html('report.html')."
        assert result[0]["graph_id"] == "test:1.0"
        assert "Relevant implementation" in result[0]["context"]
        assert result[0]["sources"] == ["test:1.0:TestClass"]
        assert result[0]["candidate_count"] == 3
        assert result[0]["selected_count"] == 1
        assert len(result[0]["top_matches"]) == 1
        assert result[0]["top_matches"][0]["name"] == "TestClass"
        assert result[0]["top_matches"][0]["type"] == "Class"
        assert result[0]["top_matches"][0]["summary"] == "Test docstring"
        assert result[0]["top_matches"][0]["id"] == "test:1.0:TestClass"


@pytest.mark.asyncio
async def test_execute_retrieve_falls_back_to_direct_retrieval_for_known_graph():
    input_data = RetrieveInput(query="data drift report preset", graph_id="evidently:0.7.21", top_k=5)

    with patch("graph_rag.mcp.tools.build_agent_graph") as mock_build_agent, \
         patch("graph_rag.mcp.tools._execute_direct_retrieve") as mock_direct:
        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "final_answer": "Based on the provided retrieved context, there is no information available.",
            "retrieved_contexts": ["No relevant nodes found."],
            "selected_ids": [],
            "candidates": [],
        }
        mock_build_agent.return_value = mock_app
        mock_direct.return_value = {
            "final_answer": "Retrieved relevant implementation context directly from the graph.",
            "retrieved_contexts": ["Graph: evidently:0.7.21\nTop matches:\n- [Class] DataDriftPreset"],
            "selected_ids": ["evidently:0.7.21:DataDriftPreset"],
            "candidates": [
                {
                    "id": "evidently:0.7.21:DataDriftPreset",
                    "name": "DataDriftPreset",
                    "type": "Class",
                    "summary": "Preset for data drift reports.",
                }
            ],
        }

        result = await execute_retrieve(input_data)

        mock_direct.assert_called_once_with(input_data)
        assert result[0]["answer"] == "Retrieved relevant implementation context directly from the graph."
        assert result[0]["sources"] == ["evidently:0.7.21:DataDriftPreset"]
        assert result[0]["candidate_count"] == 1
        assert result[0]["selected_count"] == 1
        assert result[0]["top_matches"][0]["name"] == "DataDriftPreset"

@pytest.mark.asyncio
async def test_execute_list_graphs(tmp_path):
    input_data = ListGraphsInput()

    yaml_path = tmp_path / "available_graphs.yaml"
    yaml_path.write_text('''
graphs:
  - name: "test-lib"
    versions: ["1.0"]
    latest: "1.0"
    status: "active"
''', encoding="utf-8")

    with patch("graph_rag.mcp.tools.resolve_available_graphs_path", return_value=yaml_path):
        result = await execute_list_graphs(input_data)

        assert len(result) == 1
        assert result[0]["name"] == "test-lib"
        assert result[0]["latest"] == "1.0"
        assert result[0]["graph_ids"] == ["test-lib:1.0"]


@pytest.mark.asyncio
async def test_execute_list_graphs_prefers_neo4j(monkeypatch):
    input_data = ListGraphsInput()

    class FakeResult:
        def __iter__(self):
            return iter(
                [
                    {"name": "evidently", "versions": ["0.7.21", "0.7.20"]},
                    {"name": "scikit-learn", "versions": ["1.5.2"]},
                ]
            )

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, query):
            return FakeResult()

    class FakeDriver:
        def session(self, database):
            return FakeSession()

        def close(self):
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri, auth, notifications_min_severity):
            return FakeDriver()

    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setattr("graph_rag.mcp.tools.GraphDatabase", FakeGraphDatabase)

    result = await execute_list_graphs(input_data)

    assert result == [
        {
            "name": "evidently",
            "latest": "0.7.21",
            "graph_ids": ["evidently:0.7.20", "evidently:0.7.21"],
        },
        {
            "name": "scikit-learn",
            "latest": "1.5.2",
            "graph_ids": ["scikit-learn:1.5.2"],
        },
    ]


def test_register_graph_in_config_uses_shared_registry_path(monkeypatch, tmp_path):
    yaml_path = tmp_path / "available_graphs.yaml"
    monkeypatch.setenv("GRAPH_RAG_AVAILABLE_GRAPHS", str(yaml_path))

    register_graph_in_config("evidently", "0.7.21")

    graphs = load_available_graphs()
    assert graphs == [
        {
            "name": "evidently",
            "versions": ["0.7.21"],
            "latest": "0.7.21",
            "status": "active",
        }
    ]
