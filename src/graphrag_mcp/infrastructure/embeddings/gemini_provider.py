"""Gemini embedding provider."""

from __future__ import annotations

import json
from urllib import request
from urllib.error import HTTPError, URLError

from graphrag_mcp.application.ports.parser_port import ParsedGraph
from graphrag_mcp.domain.exceptions import EmbeddingProviderError


class GeminiEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = "text-embedding-004",
        timeout_seconds: float = 30.0,
        schema_version: str = "v1",
    ) -> None:
        self.provider_name = "gemini"
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.schema_version = schema_version
        self.dimensions = 768

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
        if not self.api_key:
            raise EmbeddingProviderError(
                code="EMBEDDING_PROVIDER_ERROR",
                message="Gemini API key is not configured.",
                hint="Set the GRAPHRAG_EMBEDDING_API_KEY or GEMINI_API_KEY environment variable.",
            )

        model = self.model_name
        if not model.startswith("models/"):
            model = f"models/{model}"

        payload = json.dumps(
            {
                "model": model,
                "content": {
                    "parts": [
                        {"text": text}
                    ]
                }
            }
        ).encode("utf-8")

        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={self.api_key}"
        req = request.Request(
            url=url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                err_body = json.loads(exc.read().decode("utf-8"))
                err_msg = err_body.get("error", {}).get("message", exc.reason)
            except Exception:
                err_msg = exc.reason
            raise EmbeddingProviderError(
                code="EMBEDDING_PROVIDER_ERROR",
                message=f"Gemini embedding request failed with HTTP {exc.code}: {err_msg}.",
                hint="Verify your Gemini API key, billing status, and network connection.",
            ) from exc
        except URLError as exc:
            raise EmbeddingProviderError(
                code="EMBEDDING_PROVIDER_ERROR",
                message="Could not connect to Gemini API.",
                hint="Verify your internet connection and proxy settings.",
            ) from exc

        embedding = body.get("embedding")
        if not isinstance(embedding, dict) or "values" not in embedding:
            raise EmbeddingProviderError(
                code="EMBEDDING_PROVIDER_ERROR",
                message="Gemini returned an invalid embedding payload.",
                hint="Verify that the configured model supports embeddings.",
            )
        values = embedding.get("values")
        if not isinstance(values, list):
            raise EmbeddingProviderError(
                code="EMBEDDING_PROVIDER_ERROR",
                message="Gemini embedding values are missing or malformed.",
                hint="Verify the response structure from Gemini API.",
            )
        return [float(value) for value in values]


def _compose_embedding_text(node) -> str:
    parts = [
        node.kind,
        node.qualified_name,
        node.summary or "",
        node.signature or "",
        node.docstring or "",
    ]
    return " | ".join(part for part in parts if part).strip()


