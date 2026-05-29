"""List graphs use case."""

from graphrag_mcp.application.dto.list_graphs import GraphCatalogItem, ListGraphsRequest, ListGraphsResponse
from graphrag_mcp.application.ports.graph_repository import GraphRepository


class ListGraphsUseCase:
    def __init__(self, *, repository: GraphRepository) -> None:
        self._repository = repository

    def execute(self, request: ListGraphsRequest) -> ListGraphsResponse:
        items = self._repository.list_graphs(
            name_prefix=request.name_prefix,
            status=request.status,
            limit=request.limit,
        )
        graphs = [GraphCatalogItem(**item) for item in items]
        return ListGraphsResponse(graphs=graphs, count=len(graphs))
