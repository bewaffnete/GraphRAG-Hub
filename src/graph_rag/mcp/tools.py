"""Implementation logic for MCP tools, bridging to Graph RAG core functionality."""

from pathlib import Path
import yaml

from .schemas import RetrieveInput, ListGraphsInput

from graph_rag.agent_workflow import build_agent_graph

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_available_graphs_path() -> Path:
    """Locate available_graphs.yaml regardless of the MCP server working directory."""
    candidates = [
        Path.cwd() / "available_graphs.yaml",
        PROJECT_ROOT / "available_graphs.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def prioritize_top_matches(matches: list[dict]) -> list[dict]:
    """Sort and deduplicate selected matches for cleaner MCP output."""
    priority = {
        "Function": 0,
        "Class": 1,
        "Module": 2,
        "Example": 3,
    }
    unique: dict[str, dict] = {}
    for match in matches:
        match_id = str(match.get("id") or "").strip()
        if not match_id or match_id in unique:
            continue
        unique[match_id] = match
    return sorted(
        unique.values(),
        key=lambda item: (
            priority.get(str(item.get("type") or "Unknown"), 9),
            str(item.get("name") or ""),
        ),
    )


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


async def execute_retrieve(input_data: RetrieveInput) -> list[dict]:
    """Execute the same agentic retrieval pipeline used by `graph-rag chat`."""
    app = build_agent_graph()
    result = app.invoke(
        {
            "query": input_data.query,
            "candidates": [],
            "selected_ids": [],
            "retrieved_contexts": [],
            "final_answer": "",
        }
    )

    retrieved_contexts = result.get("retrieved_contexts", []) or []
    context = "\n\n".join(retrieved_contexts).strip()
    candidates = result.get("candidates", []) or []
    selected_ids = dedupe_ids(result.get("selected_ids", []) or [])
    selected_set = set(selected_ids)
    top_matches = prioritize_top_matches([
        candidate
        for candidate in candidates
        if candidate.get("id") in selected_set
    ])

    return [
        {
            "query": input_data.query,
            "answer": result.get("final_answer", ""),
            "context": context,
            "sources": selected_ids,
            "top_matches": top_matches,
            "candidate_count": len(candidates),
            "selected_count": len(selected_ids),
            "graph_id": input_data.graph_id,
        }
    ]


async def execute_list_graphs(input_data: ListGraphsInput) -> list[dict]:
    """Fetch a compact list of indexed libraries from the configuration file."""
    yaml_path = resolve_available_graphs_path()
    if not yaml_path.exists():
        return []

    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not content or "graphs" not in content:
            return []

        graphs = []
        for g in content.get("graphs", []):
            name = g.get("name", "")
            versions = g.get("versions", []) or []
            graph_ids = [f"{name}:{version}" for version in versions if name and version]
            graphs.append({
                "name": name,
                "latest": g.get("latest", ""),
                "graph_ids": graph_ids,
            })
        return graphs
    except Exception as e:
        print(f"Failed to read available_graphs.yaml: {e}")
        return []
