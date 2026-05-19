import pytest
from unittest.mock import patch, MagicMock

from graph_rag.mcp.schemas import RetrieveInput, ChatInput, ListGraphsInput, IngestInput
from graph_rag.mcp.tools import execute_retrieve, execute_chat, execute_list_graphs, execute_ingest

@pytest.mark.asyncio
async def test_execute_retrieve():
    input_data = RetrieveInput(query="test query", graph_id="test:1.0", top_k=5)
    
    with patch("graph_rag.mcp.tools.Neo4jGraphRetriever") as MockRetriever:
        instance = MockRetriever.return_value
        # Mock result
        mock_result = MagicMock()
        mock_node = MagicMock()
        mock_node.name = "TestClass"
        mock_node.labels = ["Class"]
        mock_node.properties = {"docstring": "Test docstring", "signature": "TestClass()"}
        mock_node.graph_id = "test:1.0:TestClass"
        mock_result.nodes = [mock_node]
        instance.retrieve.return_value = mock_result
        
        result = await execute_retrieve(input_data)
        
        assert len(result) == 1
        assert result[0]["name"] == "TestClass"
        assert result[0]["type"] == "Class"
        assert result[0]["docstring"] == "Test docstring"
        assert result[0]["signature"] == "TestClass()"
        assert result[0]["graph_id"] == "test:1.0:TestClass"

@pytest.mark.asyncio
async def test_execute_chat():
    input_data = ChatInput(query="how to use TestClass?", graph_id="test:1.0")
    
    with patch("graph_rag.mcp.tools.build_agent_graph") as mock_build_agent:
        mock_app = MagicMock()
        mock_app.invoke.return_value = {
            "final_answer": "You can use TestClass by instantiating it.",
            "selected_ids": ["test:1.0:TestClass"]
        }
        mock_build_agent.return_value = mock_app
        
        result = await execute_chat(input_data)
        
        assert result["answer"] == "You can use TestClass by instantiating it."
        assert result["sources"] == ["test:1.0:TestClass"]

@pytest.mark.asyncio
async def test_execute_list_graphs(tmp_path):
    input_data = ListGraphsInput()
    
    with patch("graph_rag.mcp.tools.Path") as MockPath:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.read_text.return_value = '''
graphs:
  - name: "test-lib"
    versions: ["1.0"]
    latest: "1.0"
    status: "active"
'''
        MockPath.return_value = mock_path_instance
        
        result = await execute_list_graphs(input_data)
        
        assert len(result) == 1
        assert result[0]["name"] == "test-lib"
        assert result[0]["versions"] == ["1.0"]

@pytest.mark.asyncio
async def test_execute_ingest():
    input_data = IngestInput(source="test-lib", graph_id="test-lib:1.0", provider="hash")
    
    with patch("graph_rag.mcp.tools.parse_python_library") as mock_parse, \
         patch("graph_rag.mcp.tools.load_snapshot_to_neo4j") as mock_load, \
         patch("graph_rag.mcp.tools.register_graph_in_config") as mock_register, \
         patch("graph_rag.mcp.tools.embed_graph") as mock_embed:
             
        mock_snapshot = MagicMock()
        mock_snapshot.metadata.name = "test-lib"
        mock_snapshot.metadata.version = "1.0"
        mock_parse.return_value = mock_snapshot
        
        mock_load.return_value = {"nodes_created": 10, "graph_id": "test-lib:1.0"}
        mock_embed.return_value = {"embedded_nodes": 10, "graph_id": "test-lib:1.0"}
        
        result = await execute_ingest(input_data)
        
        assert result["status"] == "ok"
        assert result["nodes_created"] == 10
        assert result["graph_id"] == "test-lib:1.0"
        mock_parse.assert_called_once()
        mock_load.assert_called_once()
        mock_embed.assert_called_once()
