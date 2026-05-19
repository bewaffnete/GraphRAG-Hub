import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from .schemas import RetrieveInput, ChatInput, ListGraphsInput
from .tools import execute_retrieve, execute_chat, execute_list_graphs

server = Server("graphrag-hub")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="graphrag_retrieve",
            description="Search the knowledge graph for API details. Provide a concise, factual query focusing on technical names, patterns, or behaviors (e.g., 'class definition', 'method usage', or 'error handling') without conversational filler.",
            inputSchema=RetrieveInput.model_json_schema()
        ),
        types.Tool(
            name="graphrag_chat",
            description="Ask a multi-hop question about indexed libraries. Provide a concise, factual query summarizing the technical goal (e.g., 'data processing pipeline' or 'authentication flow') without conversational filler or question marks.",
            inputSchema=ChatInput.model_json_schema()
        ),
        types.Tool(
            name="graphrag_list_graphs",
            description="List all indexed libraries with their versions and status.",
            inputSchema=ListGraphsInput.model_json_schema()
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "graphrag_retrieve":
        input_data = RetrieveInput(**arguments)
        result = await execute_retrieve(input_data)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    elif name == "graphrag_chat":
        input_data = ChatInput(**arguments)
        result = await execute_chat(input_data)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    elif name == "graphrag_list_graphs":
        input_data = ListGraphsInput(**(arguments or {}))
        result = await execute_list_graphs(input_data)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    else:
        raise ValueError(f"Unknown tool: {name}")

async def async_run():
    # Use stdio_server to communicate with the client
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
