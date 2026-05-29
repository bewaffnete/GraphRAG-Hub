"""Library entity."""

from dataclasses import dataclass


@dataclass(slots=True)
class Library:
    graph_id: str
    name: str
    version: str
    status: str = "building"
