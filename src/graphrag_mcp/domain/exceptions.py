"""Domain and application-facing exception types."""

from dataclasses import dataclass


@dataclass(slots=True)
class GraphragError(Exception):
    code: str
    message: str
    hint: str | None = None

    def __str__(self) -> str:
        return self.message


class GraphNotFoundError(GraphragError):
    pass


class NodeNotFoundError(GraphragError):
    pass


class InvalidPathError(GraphragError):
    pass


class ParseFailureError(GraphragError):
    pass


class EmbeddingProviderError(GraphragError):
    pass


class StorageError(GraphragError):
    pass


class RegistryError(GraphragError):
    pass


class HealthCheckError(GraphragError):
    pass
