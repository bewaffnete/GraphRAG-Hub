from typing import List, TypedDict, Annotated, Union, Any
import operator
import os
import json
from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from .query_decomposition import QueryDecomposer, QueryDecompositionResult, SubQuery
from .retriever import Neo4jGraphRetriever, RetrievalConfig, compress_subgraph_context
from .neo4j_loader import Neo4jConfig
from .embedding_indexer import EmbeddingConfig, extract_summary
from .llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

class AgentState(TypedDict):
    """The state of the agent."""
    query: str
    decomposition: QueryDecompositionResult
    candidates: Annotated[List[dict], operator.add]
    selected_ids: List[str]
    retrieved_contexts: Annotated[List[str], operator.add]
    final_answer: str

def decompose_query_node(state: AgentState):
    """Node to decompose the user query into structured sub-queries."""
    decomposer = QueryDecomposer()
    decomposition = decomposer.decompose(state["query"])
    return {"decomposition": decomposition, "candidates": [], "retrieved_contexts": [], "selected_ids": []}

def discover_candidates_node(sub_query: SubQuery):
    """Node to discover potential candidate nodes (metadata only)."""
    neo4j_config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password")
    )
    embedding_config = EmbeddingConfig(
        provider=os.getenv("EMBEDDING_PROVIDER", "hash"),
        model=os.getenv("EMBEDDING_MODEL", "hash-embedding-v1"),
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "256"))
    )
    
    retriever = Neo4jGraphRetriever(neo4j_config, embedding_config)
    
    candidates = []
    libs = sub_query.libraries if sub_query.libraries else [None]
    
    for lib in libs:
        try:
            config = RetrievalConfig(graph_id=lib, top_k=15, max_entities=10)
            # We use the internal _hybrid_search to get candidates without full expansion yet
            with retriever.driver.session(database=retriever.neo4j_config.database) as session:
                routes = [] if config.graph_id else retriever.route_graphs(session, sub_query.query, config)
                graph_id = config.graph_id or (routes[0].graph_id if routes else None)
                if not graph_id:
                    continue
                
                query_vector = retriever.provider.embed([sub_query.query])[0]
                seeds = retriever._hybrid_search(session, graph_id, sub_query.query, query_vector, config)
                
                for seed in seeds:
                    candidates.append({
                        "id": seed.graph_id,
                        "name": seed.name,
                        "type": seed.labels[0] if seed.labels else "Unknown",
                        "summary": extract_summary(seed.properties.get("docstring"))
                    })
        except Exception as e:
            print(f"[Discovery] Error in {lib}: {e}")
            
    retriever.close()
    return {"candidates": candidates}

def filter_candidates_node(state: AgentState):
    """Node where LLM decides which candidate nodes are relevant."""
    llm = get_llm(temperature=0, json_mode=True)
    
    candidates_str = json.dumps(state["candidates"], indent=2, ensure_ascii=False)
    
    prompt = f"""You are an expert technical filter. Given a user query and a list of candidate code entities (Classes, Methods, Modules), 
identify which entities are TRULY relevant to answering the query.

User Query: {state['query']}

Candidates:
{candidates_str}

Return a JSON object with a list of 'selected_ids'.
Example: {{"selected_ids": ["id1", "id2"]}}
Only select entities that will directly help in providing a code-level answer or architectural understanding."""

    response = llm.invoke([
        SystemMessage(content="You return strictly JSON with 'selected_ids'."),
        HumanMessage(content=prompt)
    ])
    
    try:
        # Depending on the LLM, response.content might need cleaning or parsing
        data = json.loads(response.content)
        selected_ids = data.get("selected_ids", [])
    except:
        selected_ids = []
        
    print(f"\n[Agent] Selected {len(selected_ids)} nodes out of {len(state['candidates'])} candidates.")
    return {"selected_ids": selected_ids}

def fetch_details_node(state: AgentState):
    """Node to fetch full implementation details for selected nodes."""
    if not state["selected_ids"]:
        return {"retrieved_contexts": ["No relevant nodes found."]}
        
    neo4j_config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password")
    )
    embedding_config = EmbeddingConfig(
        provider=os.getenv("EMBEDDING_PROVIDER", "hash"),
        model=os.getenv("EMBEDDING_MODEL", "hash-embedding-v1"),
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "256"))
    )
    
    retriever = Neo4jGraphRetriever(neo4j_config, embedding_config)
    nodes = retriever.get_nodes_by_ids(state["selected_ids"])
    retriever.close()
    
    # We use the existing compress_subgraph_context logic to format the full content
    formatted_context = compress_subgraph_context(
        "selected_context",
        state["query"],
        nodes,
        nodes,
        [],
        12000,
        max_entities=len(nodes)
    )
    
    print(f"[Agent] Fetched full content for selected nodes ({len(formatted_context)} chars).")
    return {"retrieved_contexts": [formatted_context]}

def synthesize_answer_node(state: AgentState):
    """Node to synthesize the final answer from fetched details."""
    llm = get_llm(temperature=0, json_mode=False)
    
    context = "\n\n".join(state["retrieved_contexts"])
    
    prompt = f"""You are a helpful assistant specialized in software libraries and repositories.
Based on the following retrieved implementation details, answer the user's original query.

User Query: {state['query']}

Retrieved Context:
{context}

Provide a clear, technical, and accurate answer with code examples if relevant."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content}

def continue_to_discovery(state: AgentState):
    """Router to send sub-queries for parallel discovery."""
    return [Send("discover", sq) for sq in state["decomposition"].sub_queries]

def build_agent_graph():
    """Builds the LangGraph for the Graph Hub Agent."""
    workflow = StateGraph(AgentState)
    
    # Define nodes
    workflow.add_node("decompose", decompose_query_node)
    workflow.add_node("discover", discover_candidates_node)
    workflow.add_node("filter", filter_candidates_node)
    workflow.add_node("fetch", fetch_details_node)
    workflow.add_node("synthesize", synthesize_answer_node)
    
    # Define edges
    workflow.set_entry_point("decompose")
    workflow.add_conditional_edges("decompose", continue_to_discovery, ["discover"])
    workflow.add_edge("discover", "filter")
    workflow.add_edge("filter", "fetch")
    workflow.add_edge("fetch", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = build_agent_graph()
    print("LangGraph workflow with two-stage retrieval compiled successfully.")
