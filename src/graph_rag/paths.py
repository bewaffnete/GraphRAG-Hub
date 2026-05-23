"""Stable filesystem paths shared across CLI and MCP entry points."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_available_graphs_path() -> Path:
    """Locate the graph registry file regardless of the current working directory."""
    explicit = os.getenv("GRAPH_RAG_AVAILABLE_GRAPHS")
    if explicit:
        return Path(explicit).expanduser()

    candidates = [
        Path.cwd() / "available_graphs.yaml",
        PROJECT_ROOT / "available_graphs.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "available_graphs.yaml"
