"""OpenAI embedding provider."""

from __future__ import annotations

import json
from urllib import request
from urllib.error import HTTPError, URLError

from graphrag_mcp.application.ports.parser_port import ParsedGraph
from graphrag_mcp.domain.exceptions import EmbeddingProviderError


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = "text-embedding-3-small",
        timeout_seconds: float = 30.0,
        schema_version: str = "v1",
    ) -> None:
        self.provider_name = "openai"
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.schema_version = schema_version

    @property
    def dimensions(self) -> int:
        if "large" in self.model_name:
            return 3072
        return 1536

    @dimensions.setter
    def dimensions(self, value: int) -> None:
        pass

    def embed_graph(self, parsed_graph: ParsedGraph) -> int:
        embedded = 0
        for node in parsed_graph.nodes:
            if node.kind not in {"Module", "Class", "Function", "Example"}:
                continue
            embedding_text = _compose_embedding_text(node)
            vector = self.embed_text(embedding_text)
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
                message="OpenAI API key is not configured.",
                hint="Set the GRAPHRAG_EMBEDDING_API_KEY or OPENAI_API_KEY environment variable.",
            )

        payload = json.dumps({"model": self.model_name, "input": text}).encode("utf-8")
        req = request.Request(
            url="https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
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
                message=f"OpenAI embedding request failed with HTTP {exc.code}: {err_msg}.",
                hint="Verify your OpenAI API key, billing status, and network connection.",
            ) from exc
        except URLError as exc:
            raise EmbeddingProviderError(
                code="EMBEDDING_PROVIDER_ERROR",
                message="Could not connect to OpenAI API.",
                hint="Verify your internet connection and proxy settings.",
            ) from exc

        data = body.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise EmbeddingProviderError(
                code="EMBEDDING_PROVIDER_ERROR",
                message="OpenAI returned an invalid embedding payload.",
                hint="Verify that the configured model supports embeddings.",
            )
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list):
            raise EmbeddingProviderError(
                code="EMBEDDING_PROVIDER_ERROR",
                message="OpenAI embedding values are missing or malformed.",
                hint="Verify the response structure from OpenAI API.",
            )
        return [float(value) for value in embedding]


def _compose_embedding_text(node) -> str:
    parts = [
        node.kind,
        node.qualified_name,
        node.summary or "",
        node.signature or "",
        node.docstring or "",
    ]
    return " | ".join(part for part in parts if part).strip()


