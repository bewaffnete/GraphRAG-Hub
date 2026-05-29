"""Ollama embedding provider."""

from __future__ import annotations

import json
from collections.abc import Iterable
from urllib import request
from urllib.error import HTTPError, URLError

from graphrag_mcp.application.ports.parser_port import ParsedGraph
from graphrag_mcp.domain.exceptions import EmbeddingProviderError


class OllamaEmbeddingProvider:
    def __init__(self, *, base_url: str, model_name: str, timeout_seconds: float = 30.0, schema_version: str = "v1") -> None:
        self.provider_name = "ollama"
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._base_urls = _candidate_base_urls(self.base_url)
        self.timeout_seconds = timeout_seconds
        self.schema_version = schema_version
        self.dimensions = 0

    def embed_graph(self, parsed_graph: ParsedGraph) -> int:
        embedded = 0
        for node in parsed_graph.nodes:
            if node.kind not in {"Module", "Class", "Function", "Example"}:
                continue
            embedding_text = _compose_embedding_text(node)
            vector = self.embed_text(embedding_text)
            self.dimensions = len(vector)
            node.metadata.update(
                {
                    "embedding": vector,
                    "embedding_provider": self.provider_name,
                    "embedding_model": self.model_name,
                    "embedding_dimensions": self.dimensions,
                    "embedding_schema_version": self.schema_version,
                    "embedding_text": embedding_text,
                }
            )
            embedded += 1
        return embedded

    def embed_text(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model_name, "input": text}).encode("utf-8")
        errors: list[str] = []
        for base_url in self._base_urls:
            try:
                return self._embed_text_at_url(base_url, payload)
            except HTTPError as exc:
                raise EmbeddingProviderError(
                    code="EMBEDDING_PROVIDER_ERROR",
                    message=f"Ollama embedding request failed with HTTP {exc.code}.",
                    hint="Verify that Ollama is running and the embedding model is available.",
                ) from exc
            except URLError as exc:
                errors.append(f"{base_url}: {exc.reason}")
                continue

        tried = ", ".join(self._base_urls)
        details = "; ".join(errors)
        raise EmbeddingProviderError(
            code="EMBEDDING_PROVIDER_ERROR",
            message=f"Could not connect to Ollama. Tried: {tried}.",
            hint=f"Start Ollama locally or run the compose ollama service. Details: {details}",
        )

    def _embed_text_at_url(self, base_url: str, payload: bytes) -> list[float]:
        req = request.Request(
            url=f"{base_url}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings or not isinstance(embeddings[0], list):
            raise EmbeddingProviderError(
                code="EMBEDDING_PROVIDER_ERROR",
                message="Ollama returned an invalid embedding payload.",
                hint="Verify that the configured model supports embeddings.",
            )
        return [float(value) for value in embeddings[0]]


def _compose_embedding_text(node) -> str:
    parts = [
        node.kind,
        node.qualified_name,
        node.summary or "",
        node.signature or "",
        node.docstring or "",
    ]
    return " | ".join(part for part in parts if part).strip()


def _candidate_base_urls(configured_base_url: str) -> list[str]:
    return _unique_urls(
        [
            #
            configured_base_url,
            # "http://ollama:11434/api",
            # "http://host.docker.internal:11434/api",
            # "http://localhost:11434/api",
        ]
    )


def _unique_urls(urls: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        cleaned = url.rstrip("/")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique
