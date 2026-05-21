# 🕸️ GraphRAG-Hub

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/database-Neo4j-green.svg)](https://neo4j.com/)
[![LangChain](https://img.shields.io/badge/framework-LangChain-orange.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Your AI coding assistant is hallucinating the API.** GraphRAG-Hub fixes that.

---

## The Problem

Every AI coding assistant has a knowledge cutoff. Ask Copilot or Claude to write code using `pandas 2.x`, `LangChain 0.3`, or any library released in the last year — and you'll get confident, detailed, **wrong** answers.

They're not lying. They just don't know what changed.

**The standard fix — RAG over documentation — is too shallow.** Markdown docs miss deprecated methods, type signatures, exception chains, and call graphs. You get text. You need structure.

---

## The Solution

GraphRAG-Hub parses any Python library directly from source using **AST analysis** and builds a **queryable knowledge graph** in Neo4j — not from docs, but from the actual code.

```
pandas 2.2.0  ──AST──▶  Graph  ──Hybrid Search──▶  LangGraph Agent  ──▶  Accurate Answer
```

Your AI assistant now knows **exactly** what exists, what's deprecated, what raises what exception, and how everything connects — for any library version you point it at.

---

## Why AST beats documentation RAG

| | Docs RAG | **GraphRAG-Hub** |
|---|---|---|
| Source of truth | Markdown text | Actual source code |
| Deprecated methods | Often undocumented | Extracted from AST |
| Type signatures | Partial / outdated | Complete, deterministic |
| Call graphs | ❌ | ✅ |
| Complexity metrics | ❌ | ✅ (cyclomatic, nesting) |
| Multi-hop reasoning | ❌ | ✅ 2-hop graph traversal |
| Indexing speed | Minutes | **Seconds** |

> Research confirms: AST-derived graphs cover ~90% of code vs 64–69% for LLM-extracted approaches, and build in seconds vs hundreds of seconds. ([arxiv:2601.08773](https://arxiv.org/abs/2601.08773))

---

## What Gets Extracted

GraphRAG-Hub builds a rich, multi-layered graph from your library:

```
Library
 └── Module
      ├── Class  ──[:INHERITS]──▶  Class
      │    └── Function  ──[:HAS_PARAM]──▶  Parameter
      │                  ──[:RETURNS]──▶    Type
      │                  ──[:RAISES]──▶     Exception
      │                  ──[:CALLS]──▶      Function
      └── Function
           └── Example
```

Beyond structure, each node is enriched with:
- **Logic Skeleton** — control flow stripped of boilerplate, optimized for LLM context
- **Complexity Metrics** — cyclomatic complexity, nesting depth, branch counts
- **Call Graph** — who calls what, which constants are used

---

## ⚡ Quick Start (Docker)

The fastest way to get started is using Docker Compose. This spins up Neo4j and the GraphRAG application.

### 1. Build and Run
```bash
git clone https://github.com/bewaffnete/GraphRAG-Hub.git
cd GraphRAG-Hub
pip install -e ".[cli]"
graph-rag setup
```

The setup wizard auto-detects nearby virtual environments, generates `.env`, chooses sane defaults for Neo4j and embeddings, and can boot `docker compose up -d` for you.

### 2. Ingest a library
To index a project, place it in the `repos/` folder and run the ingest command:
```bash
# Example: Indexing a project located at ./repos/my-cool-lib
docker compose exec graph-rag-app graph-rag ingest /repos/my-cool-lib
```

### 3. Ask questions via CLI
```bash
# Precise retrieval
docker compose exec graph-rag-app graph-rag retrieve "How does the auth flow work?"

# Agentic chat
docker compose exec graph-rag-app graph-rag chat "Explain the relationship between Module and Class"
```

---

## 🛠️ Advanced Configuration

### Ollama Setup (Local vs Docker)
By default, the app looks for Ollama on your host (Mac) for best performance.
- **Local Ollama (Mac):** Just ensure Ollama is running and `.env` has `OLLAMA_BASE_URL=http://host.docker.internal:11434`.
- **Docker Ollama:** Start with `--profile with-ollama` and set `OLLAMA_BASE_URL=http://ollama:11434`.

### Agent Integration (MCP Server)
GraphRAG-Hub includes an MCP server so your IDE (Claude Code, Cursor) can query the graph.

**MCP Configuration:**
```json
{
  "mcpServers": {
    "graphrag-hub": {
      "command": "docker",
      "args": ["exec", "-i", "graphrag-app", "graph-rag-mcp"]
    }
  }
}
```

---

## Hybrid Search: Four Signals, One Answer

When you ask a question, GraphRAG-Hub doesn't just run a similarity search. It combines:

1. **Full-text search** — exact keyword matches across names and docstrings
2. **Vector embeddings** — semantic similarity (OpenAI / Gemini / Ollama)
3. **BM25** — probabilistic relevance ranking
4. **2-hop graph traversal** — follows relationships to surface connected context

The **LangGraph agent** then decomposes your query into sub-questions, retrieves context from each signal, and synthesizes a grounded answer.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     CLI (Rich)                      │
└───────────────────────┬─────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │      Parser        │  Python ast module
              │  LibrarySnapshot   │  → classes, functions,
              └─────────┬──────────┘    params, types, exceptions
                        │
              ┌─────────▼──────────┐
              │      Enricher      │  Logic skeletons
              │                    │  Complexity metrics
              └─────────┬──────────┘  Call graphs
                        │
              ┌─────────▼──────────┐
              │  Neo4j Loader      │  Idempotent upsert
              │  (versioned)       │  graph_id = lib:version
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │     Indexer        │  Vector embeddings
              │                    │  for docstrings +
              └─────────┬──────────┘  logic skeletons
                        │
              ┌─────────▼──────────┐
              │  Hybrid Retriever  │  FTS + Vector + BM25
              │                    │  + 2-hop traversal
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │  LangGraph Agent   │  Query decomposition
              │                    │  Multi-step retrieval
              └────────────────────┘  Answer synthesis
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Graph DB | [Neo4j](https://neo4j.com/) |
| Agent Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM Providers | Ollama |
| Embeddings | OpenAI · Gemini · Ollama |
| CLI | [Rich](https://github.com/Textualize/rich) · [Questionary](https://github.com/tmbo/questionary) |
| Language | Python 3.11+ |

---

## Contributing

Issues and PRs are welcome. If you're indexing a library and hit a parsing edge case, open an issue with the library name and version.

---

## License

MIT © [bewaffnete](https://github.com/bewaffnete)
