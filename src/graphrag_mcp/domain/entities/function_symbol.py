"""Function symbol entity."""

from dataclasses import dataclass


@dataclass(slots=True)
class FunctionSymbol:
    qualified_name: str
    signature: str
    is_public: bool
