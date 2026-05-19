import pytest
import json
from mcp.server import Server
from graph_rag.mcp.server import server

@pytest.mark.asyncio
async def test_server_tools_registered():
    # The server instance is already created in graph_rag.mcp.server
    
    # We can inspect the tools by calling the list_tools handler
    # mcp.server.Server stores handlers in _request_handlers or similar, but
    # typically we can test the function directly if it was decorated,
    # but the decorators in MCP register internally.
    # We can also start a test client, but a simpler way is to check the tools list 
    # using the internal mechanisms if possible, or we can just import the handler.
    from graph_rag.mcp.server import list_tools
    
    tools = await list_tools()
    tool_names = [t.name for t in tools]
    
    assert "graphrag_retrieve" in tool_names
    assert "graphrag_chat" in tool_names
    assert "graphrag_list_graphs" in tool_names
    assert "graphrag_ingest" in tool_names

@pytest.mark.asyncio
async def test_server_call_tool_unknown():
    from graph_rag.mcp.server import call_tool
    
    with pytest.raises(ValueError, match="Unknown tool: unknown_tool"):
        await call_tool("unknown_tool", {})
