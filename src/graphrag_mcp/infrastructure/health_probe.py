"""Health probes for storage and embedding dependencies."""

from __future__ import annotations

from graphrag_mcp.application.dto.health_check import HealthComponent
from graphrag_mcp.infrastructure.config.loader import SettingsBundle


class DefaultHealthProbe:
    def __init__(self, *, settings: SettingsBundle, repository, registry) -> None:
        self._settings = settings
        self._repository = repository
        self._registry = registry

    def check(self) -> dict[str, object]:
        warnings: list[str] = []
        neo4j = self._check_neo4j()
        registry = HealthComponent(status="ok", message=f"registry backend {self._settings.registry.backend} ready")
        embedding = self._check_embedding_provider(warnings)
        overall = "ok" if neo4j.status == "ok" and embedding.status in {"ok", "disabled"} else "degraded"
        return {
            "status": overall,
            "neo4j": neo4j,
            "registry": registry,
            "embedding_provider": embedding,
            "warnings": warnings,
        }

    def _check_neo4j(self) -> HealthComponent:
        if self._settings.neo4j.backend != "neo4j":
            return HealthComponent(status="disabled", message="neo4j backend is disabled")
        try:
            driver = getattr(self._repository, "_driver", None)
            if driver is None:
                return HealthComponent(status="degraded", message="neo4j driver is not initialized")
            with driver.session(database=self._settings.neo4j.database) as session:
                result = session.run("RETURN 1 AS ok").single()
            return HealthComponent(status="ok", message="connected", details={"ok": int(result["ok"])})
        except Exception as exc:
            return HealthComponent(status="degraded", message=f"neo4j unavailable: {exc}")

    def _check_embedding_provider(self, warnings: list[str]) -> HealthComponent:
        if not self._settings.embedding.enabled:
            return HealthComponent(status="disabled", message="embedding provider is disabled")
        provider = self._settings.embedding.provider
        model = self._settings.embedding.model
        if provider == "ollama":
            try:
                from graphrag_mcp.infrastructure.embeddings.ollama_provider import OllamaEmbeddingProvider

                embedding_provider = OllamaEmbeddingProvider(
                    base_url=self._settings.embedding.base_url,
                    model_name=model,
                    timeout_seconds=self._settings.embedding.timeout_seconds,
                    schema_version=self._settings.embedding.schema_version,
                )
                vector = embedding_provider.embed_text("health check")
                return HealthComponent(status="ok", message="ollama reachable", details={"provider": provider, "model": model, "dimensions": len(vector)})
            except Exception as exc:
                warnings.append(f"Embedding provider check failed: {exc}")
                return HealthComponent(status="degraded", message=f"embedding provider unavailable: {exc}", details={"provider": provider, "model": model})
        return HealthComponent(status="ok", message="embedding provider configured", details={"provider": provider, "model": model})
