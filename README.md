# GraphRAG-Hub

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/database-Neo4j-green.svg)](https://neo4j.com/)
[![LangChain](https://img.shields.io/badge/framework-LangChain-orange.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GraphRAG-Hub** is a high-performance pipeline designed to transform Python source repositories into a rich, queryable **Graph Knowledge Hub**. Unlike traditional RAG systems that treat code as flat text, GraphRAG-Hub understands the structural and semantic relationships of your codebase, enabling precise retrieval and agentic reasoning.

---

## ⚡️Key Features

*   **Deep AST Analysis**: Extracts more than just text. It captures class hierarchies, function signatures, parameters, return types, and exceptions.
*   **AST Enrichment**:
    *   **Logic Skeleton**: Strips away boilerplate and noise, keeping only the control flow (if/else, loops, calls) for better LLM understanding.
    *   **Complexity Metrics**: Computes cyclomatic complexity, nesting depth, and branch counts.
    *   **Call Graphs**: Maps internal function calls and constant usage.
*   **Multi-Source Ingestion**: Parse local directories or entire libraries into portable JSON snapshots.
*   **Hybrid Search**: Combines full-text search, vector embeddings (OpenAI, Gemini, Ollama), BM25 and graph traversal (2-hop neighborhood).
*   **Agentic Workflow**: Powered by **LangGraph**, it decomposes complex technical queries into sub-queries, discovers relevant nodes, and synthesizes answers with full implementation context.
*   **Interactive CLI**: A beautiful, `rich`-powered CLI for seamless ingestion and exploration.

---

## The Knowledge Graph

GraphRAG-Hub builds a multi-layered graph in **Neo4j** with the following schema:

-   **Nodes**: `Library`, `Module`, `Class`, `Function`, `Parameter`, `Example`, `Type`, `Exception`.
-   **Relationships**: 
    -   `(Library)-[:CONTAINS]->(Module)`
    -   `(Module)-[:CONTAINS]->(Class|Function)`
    -   `(Class)-[:INHERITS]->(Class)`
    -   `(Function)-[:HAS_PARAM]->(Parameter)`
    -   `(Function)-[:RETURNS]->(Type)`
    -   `(Function)-[:RAISES]->(Exception)`
    -   `(Any)-[:HAS_EXAMPLE]->(Example)`

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/graph-rag.git
cd graph-rag

# Install with CLI dependencies
pip install -e '.[all]'
```

### 2. Configuration

Create a `.env` file with your credentials:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Choose your embedding provider
EMBEDDING_PROVIDER=openai # or gemini, ollama
OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...
```

### 3. Interactive Ingestion

Simply run the CLI without arguments to enter interactive mode:

```bash
graph-rag
```

---

## 🛠️ Usage (CLI)

### Ingest a Repository
Parse, load into Neo4j, and generate embeddings in one go:
```bash
graph-rag ingest ./path/to/your/library --provider openai
```

### Retrieve Context
Search for specific implementation details:
```bash
graph-rag retrieve "How is the embedding indexing handled?" --graph-id my-lib:0.1.0
```

### Agentic Chat
Run a complex query through the LangGraph agent:
```bash
graph-rag chat "Explain the relationship between the parser and the neo4j loader"
```

---

## 🏗️ Architecture

1.  **Parser**: Uses Python's `ast` module to build a `LibrarySnapshot`.
2.  **Enricher**: Enhances nodes with logic skeletons and complexity metrics.
3.  **Loader**: Upserts the snapshot into Neo4j, ensuring idempotency and maintaining versioned `graph_id`s.
4.  **Indexer**: Generates vector embeddings for docstrings and logic skeletons.
5.  **Retriever**: Performs hybrid search + graph traversal to build a "compressed context" for the LLM.
6.  **Agent**: A LangGraph state machine orchestrating the multi-step retrieval and synthesis process.

---

## 🛠️ Tech Stack

-   **Language**: [Python 3.11+](https://www.python.org/)
-   **Graph DB**: [Neo4j](https://neo4j.com/)
-   **Orchestration**: [LangChain](https://github.com/langchain-ai/langchain) & [LangGraph](https://github.com/langchain-ai/langgraph)
-   **LLMs**: OpenAI, Google Gemini, Ollama (Local)
-   **CLI**: [Rich](https://github.com/Textualize/rich), [Questionary](https://github.com/tmbo/questionary)

---
