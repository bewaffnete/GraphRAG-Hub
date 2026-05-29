"""Health check use case."""

from graphrag_mcp.application.dto.health_check import HealthCheckResponse
from graphrag_mcp.application.ports.health_probe import HealthProbe


class HealthCheckUseCase:
    def __init__(self, *, health_probe: HealthProbe, server_version: str) -> None:
        self._health_probe = health_probe
        self._server_version = server_version

    def execute(self) -> HealthCheckResponse:
        result = self._health_probe.check()
        return HealthCheckResponse(
            status=str(result["status"]),
            neo4j=result["neo4j"],
            registry=result["registry"],
            embedding_provider=result["embedding_provider"],
            server_version=self._server_version,
            warnings=list(result.get("warnings", [])),
        )
