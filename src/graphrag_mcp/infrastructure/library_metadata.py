"""Filesystem-backed library metadata resolver."""

from pathlib import Path

from graphrag_mcp.infrastructure.parsing.metadata_extractor import detect_library_identity


class FilesystemLibraryMetadataResolver:
    def resolve(self, root_path: Path) -> tuple[str | None, str | None]:
        return detect_library_identity(root_path)
