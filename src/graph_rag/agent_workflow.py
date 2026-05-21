"""LangGraph-based workflow for multi-stage graph retrieval and answer synthesis."""

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


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate candidate nodes by id while preserving useful summaries."""
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "").strip()
        if not candidate_id:
            continue
        existing = unique.get(candidate_id)
        if existing is None:
            unique[candidate_id] = dict(candidate)
            continue
        if not existing.get("summary") and candidate.get("summary"):
            existing["summary"] = candidate["summary"]
    return list(unique.values())


def dedupe_ids(values: list[str]) -> list[str]:
    """Deduplicate ids while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result

class AgentState(TypedDict):
    """
    The state maintained throughout the agentic workflow.

    Attributes:
        query (str): The original user query.
        decomposition (QueryDecompositionResult): Structured sub-queries.
        candidates (List[dict]): Metadata of nodes discovered in the first stage.
        selected_ids (List[str]): Graph IDs chosen for full detail retrieval.
        retrieved_contexts (List[str]): Formatted technical context for the final LLM.
        final_answer (str): The final synthesized answer.
    """
    query: str
    decomposition: QueryDecompositionResult
    candidates: Annotated[List[dict], operator.add]
    selected_ids: List[str]
    retrieved_contexts: Annotated[List[str], operator.add]
    final_answer: str

def decompose_query_node(state: AgentState):
    """
    Node to decompose the user query into structured sub-queries.

    Uses an LLM to break down a complex question into smaller, targeted
    searches against specific libraries.
    """
    decomposer = QueryDecomposer()
    decomposition = decomposer.decompose(state["query"])
    return {"decomposition": decomposition, "candidates": [], "retrieved_contexts": [], "selected_ids": []}

def discover_candidates_node(sub_query: SubQuery):
    """
    Node to discover potential candidate nodes (metadata only).

    Performs a lightweight hybrid search for each sub-query to identify
    likely relevant entities without fetching full implementation details.
    """
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
            with retriever.driver.session(database=retriever.neo4j_config.database) as session:
                routes = [] if config.graph_id else retriever.route_graphs(session, sub_query.query, config)
                graph_id = config.graph_id or (routes[0].graph_id if routes else None)
                
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
    """
    Node where LLM decides which candidate nodes are relevant.

    Inspects the metadata of discovered candidates and selects a subset
    that is truly critical for answering the query.
    """
    llm = get_llm(temperature=0, json_mode=True)
    unique_candidates = dedupe_candidates(state["candidates"])
    candidates_str = json.dumps(unique_candidates, indent=2, ensure_ascii=False)
    
    prompt = f"""You are an expert technical auditor tasked with selecting the most relevant code entities for a RAG system.
Given a user query and a list of candidate entities (Classes, Methods, Modules), identify the ones that are CRITICAL or highly supportive for answering the query.

### SELECTION CRITERIA:
1. **Direct Relevance**: Entities that directly address the user's question.
2. **Structural Context**: Classes or modules that house the primary logic.
3. **Dependency Context**: Methods or examples that illustrate how to use the primary entities.

User Query: {state['query']}

### Candidates:
{candidates_str}

### Instruction:
Return a JSON object with a key "selected_ids" containing a list of `graph_id` strings. Be inclusive if multiple entities are needed for a complete technical explanation.
Only select entities that will directly help in providing a code-level answer or architectural understanding.

Example:
{{"selected_ids": ["id1", "id2"]}}"""

    response = llm.invoke([
        SystemMessage(content="You return strictly JSON with 'selected_ids'."),
        HumanMessage(content=prompt)
    ])
    
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        selected_ids = dedupe_ids(data.get("selected_ids", []))
    except Exception as e:
        print(f"[Agent] JSON parse error: {e}. Raw content: {response.content}")
        selected_ids = []

    print(f"\n[Agent] Selected {len(selected_ids)} nodes out of {len(unique_candidates)} candidates.")
    return {"selected_ids": selected_ids, "candidates": unique_candidates}

def fetch_details_node(state: AgentState):
    """
    Node to fetch full implementation details for selected nodes.

    Retrieves full docstrings, signatures, and logic skeletons from Neo4j 
    for the IDs chosen in the filtering stage.
    """
    selected_ids = dedupe_ids(state["selected_ids"])
    if not selected_ids:
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
    nodes = retriever.get_nodes_by_ids(selected_ids)
    retriever.close()
    
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
    return {"retrieved_contexts": [formatted_context], "selected_ids": selected_ids}

def synthesize_answer_node(state: AgentState):
    """
    Node to synthesize the final answer from fetched details.

    Combines the rich technical context into a final grounded response.
    """
    llm = get_llm(temperature=0, json_mode=False)
    
    context = "\n\n".join(state["retrieved_contexts"])
    
    print("\n" + "="*50)
    print("DEBUG: DATA PROVIDED TO MODEL FOR AUGMENTATION")
    print("="*50)
    print(context)
    print("="*50 + "\n")
    
    prompt = f"""You are an expert technical assistant specialized in software architecture and repository analysis.
Your task is to answer the user's query STRICTLY based on the provided technical context retrieved from a Knowledge Graph.

### CONSTRAINTS & MANDATES:
1. **STRICT ADHERENCE**: You must prioritize and use the information provided in the "Retrieved Context" section. Do not ignore relevant technical details.
2. **TECHNICAL ACCURACY**: Use exact class names, function signatures, and implementation details as they appear in the context.
3. **CODE EXAMPLES**: Include code snippets if they are present in the context or can be accurately derived from the signatures provided.
4. **GROUNDING**: If the provided context is insufficient to fully answer the query, provide the best possible answer based ONLY on what is available, and explicitly state what specific information was missing from the retrieval.
5. **NO HALLUCINATION**: Do not invent functions, classes, or behaviors that are not supported by the retrieved data.

User Query: {state['query']}

### Retrieved Context:
{context}

### Final Technical Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content}

def continue_to_discovery(state: AgentState):
    """Router to send sub-queries for parallel discovery."""
    return [Send("discover", sq) for sq in state["decomposition"].sub_queries]

def build_agent_graph():
    """Builds the LangGraph for the Graph Hub Agent."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("decompose", decompose_query_node)
    workflow.add_node("discover", discover_candidates_node)
    workflow.add_node("filter", filter_candidates_node)
    workflow.add_node("fetch", fetch_details_node)
    workflow.add_node("synthesize", synthesize_answer_node)
    
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
