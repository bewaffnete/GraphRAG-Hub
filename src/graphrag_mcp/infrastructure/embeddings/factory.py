"""Embedding provider factory."""

from graphrag_mcp.infrastructure.config.settings import EmbeddingSettings
from graphrag_mcp.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider
from graphrag_mcp.infrastructure.embeddings.hash_provider import HashEmbeddingProvider
from graphrag_mcp.infrastructure.embeddings.ollama_provider import OllamaEmbeddingProvider
from graphrag_mcp.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider


def create_embedding_provider(settings: EmbeddingSettings):
    if settings.provider == "hash":
        return HashEmbeddingProvider()
    if settings.provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.base_url,
            model_name=settings.model,
            timeout_seconds=settings.timeout_seconds,
            schema_version=settings.schema_version,
        )
    if settings.provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.api_key,
            model_name=settings.model,
            timeout_seconds=settings.timeout_seconds,
            schema_version=settings.schema_version,
        )
    if settings.provider == "gemini":
        return GeminiEmbeddingProvider(
            api_key=settings.api_key,
            model_name=settings.model,
            timeout_seconds=settings.timeout_seconds,
            schema_version=settings.schema_version,
        )
    raise ValueError(f"Unsupported embedding provider: {settings.provider}")
