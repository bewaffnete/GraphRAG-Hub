# Agent Context: Graph-RAG Project

This document provides a technical overview and architectural map for AI agents interacting with this codebase.

## Project Purpose
A Graph-based Retrieval-Augmented Generation (RAG) system specialized in software engineering. It parses Python source code into an Abstract Syntax Tree (AST), enriches it with metadata, and stores it in a Neo4j graph database to allow for deep, context-aware retrieval.

## Core Architecture

### 1. Ingestion Pipeline (`src/graph_rag/`)
- **`parser.py`**: Extracts structure from Python files.
- **`ast_enrichment.py`**: Enriches raw AST data with logic skeletons, complexity metrics, and call graphs.
- **`metadata.py`**: Standardizes library-level metadata.
- **`neo4j_loader.py`**: Handles schema creation and upserting data into Neo4j using a specific naming convention for `graph_id` (`library:version`).

### 2. Retrieval Engine (`src/graph_rag/retriever.py`)
- **Hybrid Search**: Combines Vector Search (via configurable providers), Keyword Search (Neo4j Fulltext), and BM25 reranking.
- **Cross-Graph Fallback**: If no specific `graph_id` is identified via routing, the retriever falls back to searching across the entire database.
- **Sub-graph Expansion**: Traverses relationships (like `HAS_METHOD`, `INHERITS_FROM`) to gather surrounding context.
- **Context Compression**: Formats retrieved nodes into a compact text representation for LLM consumption.

### 3. Agentic Workflow (`src/graph_rag/agent_workflow.py`)
Uses `LangGraph` to orchestrate multi-step reasoning:
- **Decomposition**: Breaks complex queries into sub-queries.
- **Discovery**: Identifies candidate nodes in the graph.
- **Filtering**: Selects the most relevant nodes based on LLM reasoning.
- **Synthesis**: Generates the final answer using the augmented context.

## Graph Schema Key Labels
- `Library`: Root node for a project.
- `Module`: Represents a `.py` file.
- `Class`: Python class with inheritance links.
- `Function`: Methods and functions, including signatures and logic skeletons.
- `Example`: Code snippets found in docstrings or READMEs.

## Operational Notes for Agents
- **Debugging**: The system prints the full augmented context sent to the LLM in the `synthesize_answer_node` for transparency.
- **Graph IDs**: Nodes use a unique `graph_id` string. The format is typically `prefix:type:qualname`.
- **Environment**: Requires a running Neo4j instance and environment variables for LLM/Embedding providers (see `.env.example`).

## Key Entry Points
- `src/graph_rag/cli.py`: Main entry point for the command-line interface.
- `src/graph_rag/agent_workflow.py`: Logic for the "chat" command.
- `src/graph_rag/retriever.py`: Logic for the "retrieve" command.
