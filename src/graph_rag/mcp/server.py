"""Model Context Protocol (MCP) server implementation for Graph RAG."""

import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from .schemas import RetrieveInput, ListGraphsInput
from .tools import execute_retrieve, execute_list_graphs

server = Server("graphrag-hub")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available MCP tools for graph retrieval and indexed library discovery."""
    return [
        types.Tool(
            name="graphrag_retrieve",
            description="Search the knowledge graph and return deterministic evidence for the calling model to inspect. Provide a concise, factual query focusing on technical names, patterns, or behaviors (e.g., 'class definition', 'method usage', or 'error handling') without conversational filler.",
            inputSchema=RetrieveInput.model_json_schema()
        ),
        types.Tool(
            name="graphrag_list_graphs",
            description="List indexed libraries by name with their available graph ids.",
            inputSchema=ListGraphsInput.model_json_schema()
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle MCP tool execution requests."""
    if name == "graphrag_retrieve":
        input_data = RetrieveInput(**arguments)
        result = await execute_retrieve(input_data)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "graphrag_list_graphs":
        input_data = ListGraphsInput(**(arguments or {}))
        result = await execute_list_graphs(input_data)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    else:
        raise ValueError(f"Unknown tool: {name}")

async def async_run():
    """Run the MCP server using stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

def run():
    """Entry point for the graph-rag-mcp script."""
    asyncio.run(async_run())

if __name__ == "__main__":
    run()
