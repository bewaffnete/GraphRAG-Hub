"""Helpers for loading Graph RAG environment variables predictably."""

from __future__ import annotations

import os
from pathlib import Path


def load_app_env() -> None:
    """Load a nearby .env file so CLI commands behave the same in Docker and locally."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    candidates: list[Path] = []
    explicit = os.getenv("GRAPH_RAG_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.append(Path.cwd() / ".env")
    candidates.append(Path(__file__).resolve().parents[2] / ".env")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return

    load_dotenv(override=False)
