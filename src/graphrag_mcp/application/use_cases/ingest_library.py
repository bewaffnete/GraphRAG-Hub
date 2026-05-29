"""Ingestion use case."""

import os
from pathlib import Path
import time

from graphrag_mcp.application.dto.ingest import EmbeddingSummary, IngestLibraryRequest, IngestLibraryResponse
from graphrag_mcp.application.ports.embedding_provider import EmbeddingProvider
from graphrag_mcp.application.ports.graph_registry import GraphRegistry
from graphrag_mcp.application.ports.graph_repository import GraphRepository
from graphrag_mcp.application.ports.library_metadata_resolver import LibraryMetadataResolver
from graphrag_mcp.application.ports.parser_port import ParserPort
from graphrag_mcp.domain.exceptions import InvalidPathError
from graphrag_mcp.domain.services.graph_identity_policy import GraphIdentityPolicy


class IngestLibraryUseCase:
    def __init__(
        self,
        *,
        parser: ParserPort,
        repository: GraphRepository,
        registry: GraphRegistry,
        identity_policy: GraphIdentityPolicy,
        embedding_provider: EmbeddingProvider | None = None,
        metadata_resolver: LibraryMetadataResolver | None = None,
    ) -> None:
        self._parser = parser
        self._repository = repository
        self._registry = registry
        self._identity_policy = identity_policy
        self._embedding_provider = embedding_provider
        self._metadata_resolver = metadata_resolver

    def execute(self, request: IngestLibraryRequest) -> IngestLibraryResponse:
        started = time.perf_counter()
        root_path = _resolve_library_root_path(request.path, package_name=request.library_name)
        if not root_path.exists():
            raise InvalidPathError(
                code="PATH_NOT_FOUND",
                message=f"Library path does not exist: {root_path}",
                hint="Provide a valid path, or a package name available under GRAPHRAG_VENV_LIBS_CONTAINER_PATH.",
            )

        detected_name: str | None = None
        detected_version: str | None = None
        if self._metadata_resolver is not None:
            detected_name, detected_version = self._metadata_resolver.resolve(root_path)
        library_name = request.library_name or detected_name or root_path.name
        version = request.version or detected_version or "unknown"
        graph_id = self._identity_policy.build_graph_id(library_name, version)

        stages = ["source_discovery", "parse", "filter_to_public_api", "enrich", "map_to_graph"]
        parsed_graph = self._parser.parse_public_api(
            root_path=root_path,
            library_name=library_name,
            version=version,
            graph_id=graph_id,
        )

        embedded_nodes = 0
        provider_name: str | None = None
        model_name: str | None = None
        embedding_enabled = request.embedding_mode != "disabled" and request.ingest_mode == "parse_load_embed"
        if embedding_enabled and self._embedding_provider is not None:
            embedded_nodes = self._embedding_provider.embed_graph(parsed_graph)
            provider_name = self._embedding_provider.provider_name
            model_name = self._embedding_provider.model_name
            stages.append("embed")
        elif request.embedding_mode != "disabled" and self._embedding_provider is None:
            parsed_graph.warnings.append("Embedding was requested but no embedding provider is configured.")

        if request.ingest_mode != "parse_only":
            self._repository.store_library_graph(parsed_graph)
            stages.append("persist")

        self._registry.register_graph(
            graph_id,
            {
                "graph_id": graph_id,
                "name": library_name,
                "version": version,
                "status": "active",
                "counts": _count_kinds(parsed_graph),
            },
        )
        stages.append("registry_update")

        duration_ms = int((time.perf_counter() - started) * 1000)
        return IngestLibraryResponse(
            graph_id=graph_id,
            library_name=library_name,
            version=version,
            counts=_count_kinds(parsed_graph),
            embedding_summary=EmbeddingSummary(
                enabled=embedding_enabled and self._embedding_provider is not None,
                embedded_nodes=embedded_nodes,
                provider=provider_name,
                model=model_name,
            ),
            executed_stages=stages,
            duration_ms=duration_ms,
            warnings=list(parsed_graph.warnings),
        )


def _count_kinds(parsed_graph) -> dict[str, int]:
    counts = {
        "modules": 0,
        "classes": 0,
        "functions": 0,
        "parameters": 0,
        "examples": 0,
        "exceptions": 0,
        "returns": 0,
    }
    for node in parsed_graph.nodes:
        if node.kind == "Module":
            counts["modules"] += 1
        elif node.kind == "Class":
            counts["classes"] += 1
        elif node.kind == "Function":
            counts["functions"] += 1
        elif node.kind == "Parameter":
            counts["parameters"] += 1
        elif node.kind == "Example":
            counts["examples"] += 1
        elif node.kind == "Exception":
            counts["exceptions"] += 1
        elif node.kind == "ReturnType":
            counts["returns"] += 1
    return counts


def _resolve_library_root_path(request_path: str, *, package_name: str | None = None) -> Path:
    raw_path = Path(request_path).expanduser()
    if package_name and raw_path.exists() and _looks_like_library_search_root(raw_path):
        found = _find_package_under_roots(package_name, [raw_path])
        if found is not None:
            return found

    if raw_path.exists():
        return raw_path.resolve()

    if raw_path.is_absolute():
        return raw_path.resolve()

    found = _find_package_under_user_venv(package_name or request_path)
    if found is not None:
        return found

    return raw_path.resolve()


def _find_package_under_user_venv(package_name: str) -> Path | None:
    return _find_package_under_roots(package_name, _user_venv_search_roots())


def _find_package_under_roots(package_name: str, roots: list[Path]) -> Path | None:
    normalized_name = _normalize_package_name(package_name)
    for root in _expand_search_roots(roots):
        direct_candidates = [
            root / package_name,
            root / package_name.replace("-", "_"),
            root / package_name.replace("_", "-"),
        ]
        for candidate in direct_candidates:
            if candidate.is_dir():
                return candidate.resolve()

        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and _normalize_package_name(child.name) == normalized_name:
                return child.resolve()
    return None


def _looks_like_library_search_root(path: Path) -> bool:
    resolved = path.resolve()
    user_roots = set(_expand_search_roots(_user_venv_search_roots()))
    if resolved in user_roots:
        return True
    return path.name == "site-packages" or any(path.glob("*.dist-info"))


def _user_venv_search_roots() -> list[Path]:
    configured = os.getenv("GRAPHRAG_VENV_LIBS_CONTAINER_PATH", "/user_venv")
    base = Path(configured).expanduser()
    roots: list[Path] = []
    if base.exists():
        roots.append(base)
        roots.extend(path for path in sorted(base.glob("lib/python*/site-packages")) if path.is_dir())
        windows_site_packages = base / "Lib" / "site-packages"
        if windows_site_packages.is_dir():
            roots.append(windows_site_packages)

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)
    return unique_roots


def _expand_search_roots(roots: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for root in roots:
        if root.exists():
            expanded.append(root)
            expanded.extend(path for path in sorted(root.glob("lib/python*/site-packages")) if path.is_dir())
            windows_site_packages = root / "Lib" / "site-packages"
            if windows_site_packages.is_dir():
                expanded.append(windows_site_packages)

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in expanded:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)
    return unique_roots


def _normalize_package_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")
