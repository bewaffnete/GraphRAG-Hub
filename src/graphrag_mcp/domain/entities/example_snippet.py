"""Example snippet entity."""

from dataclasses import dataclass


@dataclass(slots=True)
class ExampleSnippet:
    qualified_name: str
    code: str
