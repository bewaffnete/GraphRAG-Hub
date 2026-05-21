import pytest

@pytest.mark.asyncio
async def test_server_tools_registered():
    from graph_rag.mcp.server import list_tools

    tools = await list_tools()
    tool_names = [t.name for t in tools]

    assert "graphrag_retrieve" in tool_names
    assert "graphrag_list_graphs" in tool_names
    assert len(tool_names) == 2

@pytest.mark.asyncio
async def test_server_call_tool_unknown():
    from graph_rag.mcp.server import call_tool
    
    with pytest.raises(ValueError, match="Unknown tool: unknown_tool"):
        await call_tool("unknown_tool", {})
