"""Pydantic models for Model Context Protocol (MCP) tool inputs and outputs."""

from typing import Optional
from pydantic import BaseModel, Field

class RetrieveInput(BaseModel):
    """Input for the graphrag_retrieve tool."""
    query: str = Field(..., description="Concise, factual query targeting specific entities or logic (e.g. 'class definition')")
    graph_id: Optional[str] = Field(default=None, description="e.g. 'pandas:2.2.0'; if omitted, search all graphs")
    top_k: int = Field(default=5, description="Number of results to retrieve")

class ListGraphsInput(BaseModel):
    """Input for the graphrag_list_graphs tool (currently empty)."""
    pass
