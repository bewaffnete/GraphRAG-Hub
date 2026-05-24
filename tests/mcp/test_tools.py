import pytest
from unittest.mock import patch

from graph_rag.mcp.schemas import RetrieveInput, ListGraphsInput
from graph_rag.mcp.tools import execute_retrieve, execute_list_graphs
from graph_rag.query_decomposition import register_graph_in_config, load_available_graphs
from graph_rag.retriever import RetrievedNode, RetrievalResult


def _retrieval_result() -> RetrievalResult:
    seeds = [
        RetrievedNode(
            graph_id="test:1.0:TestClass",
            labels=["Class"],
            name="TestClass",
            score=1.0,
            properties={"docstring": "Test docstring"},
        ),
        RetrievedNode(
            graph_id="test:1.0:Noise",
            labels=["Example"],
            name="Noise",
            score=0.3,
            properties={"docstring": "Irrelevant"},
        ),
        RetrievedNode(
            graph_id="test:1.0:TestClass",
            labels=["Class"],
            name="TestClass",
            score=0.9,
            properties={"docstring": ""},
        ),
    ]
    nodes = [
        seeds[0],
        RetrievedNode(
            graph_id="test:1.0:helper",
            labels=["Function"],
            name="helper",
            score=0.7,
            properties={"docstring": "Helper details"},
        ),
        seeds[0],
    ]
    return RetrievalResult(
        query="test query",
        graph_id="test:1.0",
        routed_graphs=[],
        seeds=seeds,
        nodes=nodes,
        edges=[],
        compressed_context="Graph: test:1.0\nRelevant implementation:\n[Class] TestClass",
    )


class FakeRetriever:
    last_config = None
    closed = False

    def __init__(self, neo4j_config, embedding_config):
        self.neo4j_config = neo4j_config
        self.embedding_config = embedding_config

    def retrieve(self, query, config):
        FakeRetriever.last_config = config
        assert query == "test query"
        return _retrieval_result()

    def close(self):
        FakeRetriever.closed = True

@pytest.mark.asyncio
async def test_execute_retrieve():
    input_data = RetrieveInput(query="test query", graph_id="test:1.0", top_k=5)

    with patch("graph_rag.mcp.tools.Neo4jGraphRetriever", FakeRetriever):
        result = await execute_retrieve(input_data)

        assert len(result) == 1
        assert result[0]["query"] == "test query"
        assert result[0]["answer"] == "Retrieved deterministic graph evidence. Use top_matches and context to choose the useful nodes."
        assert result[0]["graph_id"] == "test:1.0"
        assert "Relevant implementation" in result[0]["context"]
        assert result[0]["sources"] == ["test:1.0:TestClass", "test:1.0:helper"]
        assert result[0]["candidate_count"] == 3
        assert result[0]["selected_count"] == 2
        assert result[0]["routed_graphs"] == []
        assert result[0]["top_matches"] == [
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
        ]
        assert FakeRetriever.last_config.graph_id == "test:1.0"
        assert FakeRetriever.last_config.top_k == 5
        assert FakeRetriever.closed is True


@pytest.mark.asyncio
async def test_execute_retrieve_returns_empty_result_when_direct_retrieval_fails():
    input_data = RetrieveInput(query="data drift report preset", graph_id="evidently:0.7.21", top_k=5)

    with patch("graph_rag.mcp.tools._execute_direct_retrieve", return_value=None) as mock_direct:
        result = await execute_retrieve(input_data)

        mock_direct.assert_called_once_with(input_data)
        assert result == [
            {
                "query": "data drift report preset",
                "answer": "No graph evidence could be retrieved.",
                "context": "",
                "sources": [],
                "top_matches": [],
                "candidate_count": 0,
                "selected_count": 0,
                "graph_id": "evidently:0.7.21",
                "routed_graphs": [],
            }
        ]

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
