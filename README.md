<div align="center">
# 🧠 Graph-RAG-MCP
 
**Give your AI assistant a precise, queryable map of any Python codebase.**
 
Graph-RAG-MCP parses Python source code into a knowledge graph, then exposes it over the [Model Context Protocol](https://modelcontextprotocol.io/) — so Claude and other AI assistants can find exactly the right API symbol, docstring, or inheritance chain on the first try.
 
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/graph-Neo4j-008CC1.svg)](https://neo4j.com/)
[![MCP](https://img.shields.io/badge/protocol-MCP-blueviolet.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
 
</div>
 
## ✨ Why Graph-RAG-MCP?
 
Standard RAG splits code into text chunks and searches by similarity — you get fragments, lose context, and the AI fills in the gaps with guesses. **Graph-RAG-MCP does it differently:**
 
| Problem with naive RAG | Graph-RAG-MCP solution |
|---|---|
| Loses class/function relationships | Stores everything as a graph with typed edges |
| Misses inheritance chains | Traverse "inherits from X" in a single query |
| Buries docstrings in noise | Surgically extracts public API surface via AST |
| Semantic drift on code symbols | Combines vector search *and* graph traversal |
 
The result: your AI assistant retrieves **surgical, context-rich evidence** instead of vague text fragments.
 
---
 
## 🚀 Quick Start
 
### Prerequisites
 
- Python 3.11+
- Docker & Docker Compose
- An embedding provider key: [Gemini](https://ai.google.dev/), [OpenAI](https://openai.com/), or a local [Ollama](https://ollama.com/) instance
### 1 — Clone & configure
 
```bash
git clone https://github.com/your-repo/graph-rag-mcp.git
cd graph-rag-mcp
 
# Copy the example env and fill in your credentials
cp .env.example .env
```
 
### 2 — Start everything
 
```bash
./scripts/compose-up.sh
```
 
This starts Neo4j and the Graph-RAG-MCP service in Docker. That's it — you're ready to ingest libraries.
 
### 3 — Connect to Claude Desktop
 
Add this to your `claude_desktop_config.json`:
 
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
 
Restart Claude Desktop and the tools will appear automatically.
 
---
 
## ⚙️ Configuration
 
All settings live in your `.env` file:
 
| Variable | Description | Default |
|---|---|---|
| `GRAPHRAG_NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `GRAPHRAG_EMBEDDING_PROVIDER` | `gemini` · `openai` · `ollama` · `hash` | `ollama` |
| `VENV_LIBS_PATH` | Path to site-packages for ingestion | `./.docker/user_venv_empty` |
 
> **Tip:** Use `hash` as the embedding provider for a zero-dependency local setup with no API keys required (lower semantic quality, great for testing).
 
---
 
## 🛠️ MCP Tools Reference
 
Once connected, your AI assistant has access to six tools:
 
| Tool | What it does |
|---|---|
| `graphrag_list_graphs` | List all ingested libraries |
| `graphrag_ingest_library` | Ingest a library from a local path or package name |
| `graphrag_retrieve_candidates` | Semantic search across code symbols |
| `graphrag_retrieve_details` | Get full docstrings, signatures & relationships for specific nodes |
| `graphrag_rebuild_embeddings` | Rebuild vector embeddings for a graph |
| `graphrag_health` | Check service and database connectivity |
 
**Example flow:**
1. `graphrag_ingest_library` → ingest `requests` or any local package
2. `graphrag_retrieve_candidates` → "how do I send a POST with auth headers?"
3. `graphrag_retrieve_details` → get the full `Session.request()` signature + docstring
---
 
## 🏗️ Architecture
 
Graph-RAG-MCP follows **Clean Architecture** — each layer has one job and one direction of dependency.
 
```
┌─────────────────────────────────────────────┐
│              Interfaces                     │
│   MCP Server (AI tools)  │  Operator CLI    │
├─────────────────────────────────────────────┤
│              Application                    │
│   IngestLibrary  │  RetrieveCandidates      │
├─────────────────────────────────────────────┤
│              Domain                         │
│   GraphNode · Library · Scoring policies    │
├─────────────────────────────────────────────┤
│              Infrastructure                 │
│  Neo4j repo · AST parser · Embeddings · YAML│
└─────────────────────────────────────────────┘
```
 
**Key design decisions:**
 
- **AST-first parsing** — only public symbols are extracted. Private implementation details never enter the graph.
- **Neo4j for storage** — enables relationship queries impossible with flat vector stores (e.g. "all subclasses of `BaseModel`").
- **Pluggable embeddings** — swap between Gemini, OpenAI, and Ollama without touching application logic.
---
 
## 💻 Development Setup
 
Prefer running natively? No Docker required:
 
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
 
graph-rag-cli                    # Launch the interactive operator CLI
```
 
### Operator CLI
 
The CLI gives you a menu-driven interface for ingestion, graph management, and exploration:
 
```bash
# Linux / macOS (via Docker)
./scripts/mcp-docker-run.sh
 
# Windows PowerShell (via Docker)
.\scripts\mcp-docker-run.ps1
```
 
### Running Tests
 
```bash
pytest
```

---
 
## 📄 License
 
MIT © [bewaffnete](https://github.com/bewaffnete)
