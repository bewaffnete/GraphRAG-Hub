import pytest
from unittest.mock import patch, MagicMock

from graph_rag.mcp.schemas import RetrieveInput, ListGraphsInput
from graph_rag.mcp.tools import execute_retrieve, execute_list_graphs

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
