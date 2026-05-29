"""DTOs for health checks."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class HealthCheckRequest:
    include_details: bool = True


@dataclass(slots=True)
class HealthComponent:
    status: str
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class HealthCheckResponse:
    status: str
    neo4j: HealthComponent
    registry: HealthComponent
    embedding_provider: HealthComponent
    server_version: str
    warnings: list[str] = field(default_factory=list)
