"""Port for detecting library metadata from a filesystem path."""

from pathlib import Path
from typing import Protocol


class LibraryMetadataResolver(Protocol):
    def resolve(self, root_path: Path) -> tuple[str | None, str | None]:
        """Return detected logical library name and version, if available."""
