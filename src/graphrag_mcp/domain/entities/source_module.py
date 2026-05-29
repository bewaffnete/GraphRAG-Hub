"""Source module entity."""

from dataclasses import dataclass


@dataclass(slots=True)
class SourceModule:
    qualified_name: str
    source_path: str
    is_public: bool
