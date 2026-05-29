"""Class symbol entity."""

from dataclasses import dataclass


@dataclass(slots=True)
class ClassSymbol:
    qualified_name: str
    is_public: bool
