"""MCP server implementation over standard input/output transport."""

import asyncio
import json

from graphrag_mcp.bootstrap.container import build_container
from graphrag_mcp.interfaces.mcp.tool_registry import build_tool_registry


class GraphragMcpServer:
    def __init__(self) -> None:
        self.container = build_container()
        self.tools = build_tool_registry(self.container)

    def call_tool(self, name: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        if name not in self.tools:
            return {
                "error": {
                    "code": "INVALID_INPUT",
                    "message": f"Unknown tool: {name}",
                    "hint": "Call one of the registered graphrag_* tools.",
                }
            }
        return self.tools[name](payload or {})


async def main() -> None:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as types

    mcp_server = GraphragMcpServer()
    server = Server("graphrag-mcp")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="graphrag_list_graphs",
                description="Return the catalog of available graphs so the calling model can choose a graph scope before retrieval.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name_prefix": {"type": "string", "description": "optional, filters by library name prefix"},
                        "status": {"type": "string", "description": "optional, filters by graph status"},
                        "limit": {"type": "integer", "description": "optional, default 50, max 500"},
                    }
                }
            ),
            types.Tool(
                name="graphrag_ingest_library",
                description="Parse the library's public API, construct a graph, persist it, and optionally build embeddings.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "required absolute/workspace path or package name available under GRAPHRAG_VENV_LIBS_CONTAINER_PATH"},
                        "library_name": {"type": "string", "description": "optional explicit logical name override"},
                        "version": {"type": "string", "description": "optional explicit version override"},
                        "ingest_mode": {"type": "string", "enum": ["parse_only", "parse_load", "parse_load_embed"], "description": "optional, default 'parse_load_embed'"},
                        "embedding_mode": {"type": "string", "enum": ["enabled", "disabled", "rebuild"], "description": "optional, default 'enabled'"},
                    },
                    "required": ["path"]
                }
            ),
            types.Tool(
                name="graphrag_retrieve_candidates",
                description="Return a compact shortlist of candidate nodes for the calling model.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "required concise technical query"},
                        "graph_id": {"type": "string", "description": "optional single graph scope"},
                        "graph_ids": {"type": "array", "items": {"type": "string"}, "description": "optional multi-graph scope"},
                        "top_k": {"type": "integer", "description": "optional, default 8"},
                        "labels": {"type": "array", "items": {"type": "string"}, "description": "optional filter on node kinds"},
                        "traversal_depth": {"type": "integer", "description": "optional, default 2"},
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="graphrag_retrieve_details",
                description="Return detailed payloads for the exact nodes selected by the caller.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_ids": {"type": "array", "items": {"type": "string"}, "description": "required list of node IDs"},
                        "include_relationships": {"type": "boolean", "description": "optional, default true"},
                        "include_docstring": {"type": "boolean", "description": "optional, default true"},
                        "include_logic_skeleton": {"type": "boolean", "description": "optional, default true"},
                        "include_source_excerpt": {"type": "boolean", "description": "optional, default false"},
                    },
                    "required": ["node_ids"]
                }
            ),
            types.Tool(
                name="graphrag_rebuild_embeddings",
                description="Rebuild embeddings for a specific graph or graph set after a schema or model change.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string", "description": "required graph ID"},
                        "provider": {"type": "string", "description": "optional provider override"},
                        "model": {"type": "string", "description": "optional model override"},
                        "schema_version": {"type": "string", "description": "optional schema version override"},
                    },
                    "required": ["graph_id"]
                }
            ),
            types.Tool(
                name="graphrag_health",
                description="Return current service health for operators and calling clients.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent]:
        try:
            result = mcp_server.call_tool(name, arguments)
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False)
                )
            ]
        except Exception as exc:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"error": {"code": "INTERNAL_ERROR", "message": str(exc)}}, indent=2, ensure_ascii=False)
                )
            ]

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            options,
        )


def main_sync() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
