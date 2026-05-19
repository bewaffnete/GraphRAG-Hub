"""Pydantic models for Model Context Protocol (MCP) tool inputs and outputs."""

from typing import Optional
from pydantic import BaseModel, Field

class RetrieveInput(BaseModel):
    """Input for the graphrag_retrieve tool."""
    query: str = Field(..., description="Concise, factual query targeting specific entities or logic (e.g. 'class definition')")
    graph_id: Optional[str] = Field(default=None, description="e.g. 'pandas:2.2.0'; if omitted, search all graphs")
    top_k: int = Field(default=5, description="Number of results to retrieve")

class ChatInput(BaseModel):
    """Input for the graphrag_chat tool."""
    query: str = Field(..., description="Concise technical statement (e.g. 'implement data processing')")
    graph_id: Optional[str] = Field(default=None, description="Optional graph_id to scope the search")

class ListGraphsInput(BaseModel):
    """Input for the graphrag_list_graphs tool (currently empty)."""
    pass

class IngestInput(BaseModel):
    """Input for the graphrag_ingest tool (planned)."""
    source: str = Field(..., description="Local path OR pip package name, e.g. 'pandas==2.2.0'")
    graph_id: str = Field(..., description="e.g. 'pandas:2.2.0'")
    provider: str = Field(..., description="Embedding provider: 'openai' | 'gemini' | 'ollama' | 'hash'")
