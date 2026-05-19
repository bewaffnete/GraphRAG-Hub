from typing import List, Optional, Any
import yaml
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .llm import get_llm

class SubQuery(BaseModel):
    id: int = Field(description="Unique identifier for the sub-query")
    query: str = Field(description="The decomposed sub-query text")
    libraries: List[str] = Field(description="List of libraries this sub-query relates to")
    entities: List[str] = Field(description="Key entities (classes, methods) involved")
    aspects: List[str] = Field(description="Specific aspects or features to search for")

class QueryDecompositionResult(BaseModel):
    reasoning: str = Field(description="Brief explanation of the decomposition logic")
    sub_queries: List[SubQuery] = Field(description="List of structured sub-queries")

def load_available_graphs(path: str = "available_graphs.yaml") -> List[dict]:
    """Loads the list of available graphs from a YAML file."""
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    if not data or 'graphs' not in data:
        return []
    return data.get('graphs', [])

def register_graph_in_config(name: str, version: str, path: str = "available_graphs.yaml"):
    """Registers or updates a graph in the available_graphs.yaml file."""
    name = name.replace("_", "-").lower()
    graphs = load_available_graphs(path)
    
    found = False
    for g in graphs:
        if g['name'] == name:
            if version not in g.get('versions', []):
                g.setdefault('versions', []).append(version)
            g['latest'] = version
            g['status'] = 'active'
            found = True
            break
            
    if not found:
        graphs.append({
            'name': name,
            'versions': [version],
            'latest': version,
            'status': 'active'
        })
        
    with open(path, 'w') as f:
        yaml.safe_dump({'graphs': graphs}, f, sort_keys=False)
    
    print(f"[Config] Registered {name} (version {version}) in {path}")

class QueryDecomposer:
    def __init__(self):
        self.llm = get_llm(temperature=0, json_mode=True)
        self.parser = JsonOutputParser(pydantic_object=QueryDecompositionResult)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at decomposing complex technical queries into structured sub-queries for a Graph-based RAG system.
Your goal is to break down the user's query into smaller, targeted searches that can be performed against specific library graphs.

### QUERY STYLE RULES:
- **Conciseness**: Keep sub-queries as short and factual as possible. Focus on key technical terms, class names, or specific behaviors.
- **No Conversational Filler**: Avoid "How can I", "How to", "Tell me about", etc.
- **No Questions**: Do not use question marks.
- **Examples**: Instead of "How can I implement a specific feature?", use "implement [feature_name]" or "[entity_name] usage".

CRITICAL RULE: You MUST ONLY use libraries that are physically present in the Graph Hub.
Available libraries:
{available_libraries}

If a query asks about a library not in this list, focus only on the available ones or explain why it cannot be decomposed further.

{format_instructions}"""),
            ("human", "{query}"),
        ])

    def decompose(self, query: str) -> QueryDecompositionResult:
        graphs = load_available_graphs()
        lib_info = "\n".join([f"- {g['name']}: {g.get('description', 'No description')}" for g in graphs])
        
        chain = self.prompt | self.llm | self.parser
        
        result_dict = chain.invoke({
            "query": query,
            "available_libraries": lib_info,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        return QueryDecompositionResult(**result_dict)

if __name__ == "__main__":
    try:
        decomposer = QueryDecomposer()
        print("QueryDecomposer initialized successfully.")
    except Exception as e:
        print(f"Initialization check: {e}")
