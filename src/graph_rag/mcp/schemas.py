from typing import Optional
from pydantic import BaseModel, Field

class RetrieveInput(BaseModel):
    query: str = Field(..., description="Natural language question")
    graph_id: Optional[str] = Field(default=None, description="e.g. 'pandas:2.2.0'; if omitted, search all graphs")
    top_k: int = Field(default=5, description="Number of results to retrieve")

class ChatInput(BaseModel):
    query: str = Field(..., description="Multi-hop question for the LangGraph agent")
    graph_id: Optional[str] = Field(default=None, description="Optional graph_id to scope the search")

class ListGraphsInput(BaseModel):
    pass

class IngestInput(BaseModel):
    source: str = Field(..., description="Local path OR pip package name, e.g. 'pandas==2.2.0'")
    graph_id: str = Field(..., description="e.g. 'pandas:2.2.0'")
    provider: str = Field(..., description="Embedding provider: 'openai' | 'gemini' | 'ollama' | 'hash'")
