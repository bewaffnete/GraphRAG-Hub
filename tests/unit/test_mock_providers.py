"""Unit tests for OpenAI and Gemini real embedding providers with mocked HTTP endpoints."""

import json
from pathlib import Path
from urllib.error import URLError
import pytest
from graphrag_mcp.application.ports.parser_port import ParsedGraph
from graphrag_mcp.domain.entities.graph_node import GraphNode
from graphrag_mcp.domain.exceptions import EmbeddingProviderError
from graphrag_mcp.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider
from graphrag_mcp.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider
from graphrag_mcp.infrastructure.embeddings.ollama_provider import OllamaEmbeddingProvider


def test_openai_embedding_provider_requires_api_key() -> None:
    provider = OpenAIEmbeddingProvider(api_key=None)
    with pytest.raises(EmbeddingProviderError) as exc_info:
        provider.embed_text("test")
    assert "OpenAI API key is not configured" in str(exc_info.value.message)


def test_openai_embedding_provider_happy_path(mocker) -> None:
    provider = OpenAIEmbeddingProvider(api_key="fake-key", model_name="text-embedding-3-small")
    assert provider.provider_name == "openai"
    assert provider.dimensions == 1536

    # Test dynamic dimensions configuration
    provider.model_name = "text-embedding-3-large"
    assert provider.dimensions == 3072

    # Mock urllib.request.urlopen for embed_text
    mock_response = mocker.MagicMock()
    mock_response.read.return_value = json.dumps({
        "object": "list",
        "data": [{"embedding": [0.5] * 3072}]
    }).encode("utf-8")
    
    mock_urlopen = mocker.patch("urllib.request.urlopen")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    vector = provider.embed_text("test content")
    assert len(vector) == 3072
    assert vector[0] == 0.5

    # Verify HTTP request content
    mock_urlopen.assert_called_once()
    args, kwargs = mock_urlopen.call_args
    req = args[0]
    assert req.get_header("Authorization") == "Bearer fake-key"
    assert req.get_header("Content-type") == "application/json"


def test_openai_embed_graph(mocker) -> None:
    provider = OpenAIEmbeddingProvider(api_key="fake-key", model_name="text-embedding-3-small")
    
    mock_response = mocker.MagicMock()
    mock_response.read.return_value = json.dumps({
        "object": "list",
        "data": [{"embedding": [0.1] * 1536}]
    }).encode("utf-8")
    
    mock_urlopen = mocker.patch("urllib.request.urlopen")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    node1 = GraphNode(
        node_id="pkg:1.0.0:Class:pkg.MyClass",
        graph_id="pkg:1.0.0",
        kind="Class",
        name="MyClass",
        display_name="MyClass",
        qualified_name="pkg.MyClass",
        source_path="pkg/mod.py",
        line_start=10,
        line_end=20,
        summary="A simple class.",
    )
    node2 = GraphNode(
        node_id="pkg:1.0.0:Parameter:pkg.MyClass:x",
        graph_id="pkg:1.0.0",
        kind="Parameter",
        name="x",
        display_name="x",
        qualified_name="pkg.MyClass:x",
        source_path="pkg/mod.py",
        line_start=12,
        line_end=12,
    )

    parsed_graph = ParsedGraph(
        library_name="pkg",
        version="1.0.0",
        root_path=Path("/fake"),
        nodes=[node1, node2],
    )

    embedded_count = provider.embed_graph(parsed_graph)
    assert embedded_count == 1  # Only eligible kind Class is embedded, Parameter is skipped.
    
    assert "embedding" in node1.metadata
    assert len(node1.metadata["embedding"]) == 1536
    assert node1.metadata["embedding_provider"] == "openai"
    assert node1.metadata["embedding_model"] == "text-embedding-3-small"
    assert node1.metadata["embedding_dimensions"] == 1536

    assert "embedding" not in node2.metadata


def test_gemini_embedding_provider_requires_api_key() -> None:
    provider = GeminiEmbeddingProvider(api_key=None)
    with pytest.raises(EmbeddingProviderError) as exc_info:
        provider.embed_text("test")
    assert "Gemini API key is not configured" in str(exc_info.value.message)


def test_gemini_embedding_provider_happy_path(mocker) -> None:
    provider = GeminiEmbeddingProvider(api_key="fake-gemini-key", model_name="text-embedding-004")
    assert provider.provider_name == "gemini"
    assert provider.dimensions == 768

    # Mock urllib.request.urlopen for embed_text
    mock_response = mocker.MagicMock()
    mock_response.read.return_value = json.dumps({
        "embedding": {"values": [0.8] * 768}
    }).encode("utf-8")
    
    mock_urlopen = mocker.patch("urllib.request.urlopen")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    vector = provider.embed_text("hello gemini")
    assert len(vector) == 768
    assert vector[0] == 0.8

    # Verify HTTP request content
    mock_urlopen.assert_called_once()
    args, kwargs = mock_urlopen.call_args
    req = args[0]
    assert "fake-gemini-key" in req.full_url
    assert req.get_header("Content-type") == "application/json"


def test_ollama_provider_falls_back_to_host_ollama(mocker) -> None:
    provider = OllamaEmbeddingProvider(
        base_url="http://ollama:11434/api",
        model_name="qwen3-embedding:0.6b",
    )
    mock_response = mocker.MagicMock()
    mock_response.read.return_value = json.dumps({"embeddings": [[0.1, 0.2, 0.3]]}).encode("utf-8")

    def fake_urlopen(req, timeout):
        if req.full_url == "http://ollama:11434/api/embed":
            raise URLError("compose ollama unavailable")
        if req.full_url == "http://host.docker.internal:11434/api/embed":
            return mock_response
        raise AssertionError(f"unexpected url: {req.full_url}")

    mock_urlopen = mocker.patch("urllib.request.urlopen", side_effect=fake_urlopen)
    mock_response.__enter__.return_value = mock_response

    assert provider.embed_text("health check") == [0.1, 0.2, 0.3]
    assert mock_urlopen.call_count == 2
