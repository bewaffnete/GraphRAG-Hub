"""Graph identity value object."""

from dataclasses import dataclass
import re


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def normalize_slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-")
    return slug or "unknown"


@dataclass(frozen=True, slots=True)
class GraphId:
    library_name: str
    version: str

    @classmethod
    def create(cls, library_name: str, version: str) -> "GraphId":
        return cls(library_name=normalize_slug(library_name), version=version.strip() or "unknown")

    def __str__(self) -> str:
        return f"{self.library_name}:{self.version}"
