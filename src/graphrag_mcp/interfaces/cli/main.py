"""Operator CLI with interactive library selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from graphrag_mcp.application.dto.ingest import IngestLibraryRequest
from graphrag_mcp.application.dto.retrieve_candidates import RetrieveCandidatesRequest
from graphrag_mcp.bootstrap.container import build_container
from graphrag_mcp.infrastructure.parsing.metadata_extractor import detect_library_identity


def main() -> int:
    parser = argparse.ArgumentParser(prog="graphrag-mcp")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("path")
    ingest_parser.add_argument("--library-name")
    ingest_parser.add_argument("--version")
    ingest_parser.add_argument("--ingest-mode", default="parse_load_embed")
    ingest_parser.add_argument("--embedding-mode", default="enabled")

    menu_parser = subparsers.add_parser("menu")
    menu_parser.add_argument("--root", default=None)
    menu_parser.add_argument("--ingest-mode", default="parse_load")
    menu_parser.add_argument("--embedding-mode", default="disabled")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--graph-id", default=None)
    search_parser.add_argument("--top-k", type=int, default=8)
    search_parser.add_argument("--label", action="append", dest="labels", default=None)

    args = parser.parse_args()
    container = build_container()

    if args.command is None:
        return _run_home_menu(container)

    if args.command == "ingest":
        response = container.ingest_library.execute(
            _build_request(
                path=args.path,
                library_name=args.library_name,
                version=args.version,
                ingest_mode=args.ingest_mode,
                embedding_mode=args.embedding_mode,
            )
        )
        _print_response(response)
        return 0

    if args.command == "menu":
        scan_root = Path(args.root).expanduser().resolve() if args.root else _default_library_scan_root()
        selected_path = _select_library_path(scan_root)
        if selected_path is None:
            print("Cancelled.")
            return 0
        library_name = _prompt_optional("Library name", default=selected_path.name)
        version = _prompt_optional("Version", default="unknown")
        response = container.ingest_library.execute(
            _build_request(
                path=str(selected_path),
                library_name=library_name,
                version=version,
                ingest_mode=args.ingest_mode,
                embedding_mode=args.embedding_mode,
            )
        )
        _print_response(response)
        return 0

    if args.command == "search":
        response = container.retrieve_candidates.execute(
            RetrieveCandidatesRequest(
                query=args.query,
                graph_id=args.graph_id,
                top_k=args.top_k,
                labels=args.labels,
            )
        )
        print(
            json.dumps(
                {
                    "query": response.query,
                    "graph_scope": {
                        "requested_graph_id": response.graph_scope.requested_graph_id,
                        "requested_graph_ids": response.graph_scope.requested_graph_ids,
                        "resolved_graph_ids": response.graph_scope.resolved_graph_ids,
                    },
                    "routed_graphs": [{"graph_id": item.graph_id, "score": item.score} for item in response.routed_graphs],
                    "candidate_count": response.candidate_count,
                    "top_matches": [
                        {
                            "node_id": item.node_id,
                            "graph_id": item.graph_id,
                            "kind": item.kind,
                            "name": item.name,
                            "qualified_name": item.qualified_name,
                            "summary": item.summary,
                            "score": item.score,
                        }
                        for item in response.top_matches
                    ],
                    "context_preview": response.context_preview,
                    "warnings": response.warnings,
                },
                indent=2,
            )
        )
        return 0

    return 1


def _run_home_menu(container) -> int:
    while True:
        print("\nGraphRAG MCP")
        print("1. Ingest a local repository")
        print("2. Search graph")
        print("3. Exit")
        choice = input("Choose an action: ").strip().lower()
        if choice == "1":
            selected_path = _select_library_path(_default_library_scan_root())
            if selected_path is None:
                continue
            library_name, version = detect_library_identity(selected_path)
            print(f"Detected library: {library_name or selected_path.name}")
            print(f"Detected version: {version or 'unknown'}")
            response = container.ingest_library.execute(
                _build_request(
                    path=str(selected_path),
                    library_name=library_name,
                    version=version,
                    ingest_mode="parse_load_embed",
                    embedding_mode="enabled",
                )
            )
            _print_response(response)
            continue
        if choice == "2":
            graph_id = input("Graph id: ").strip() or None
            query = input("Search query: ").strip()
            if not query:
                print("Query is required.")
                continue
            response = container.retrieve_candidates.execute(
                RetrieveCandidatesRequest(query=query, graph_id=graph_id, top_k=8)
            )
            _print_search_response(response)
            continue
        if choice == "3" or choice == "q":
            return 0
        print("Invalid choice.")


def _build_request(
    *,
    path: str,
    library_name: str | None,
    version: str | None,
    ingest_mode: str,
    embedding_mode: str,
) -> IngestLibraryRequest:
    return IngestLibraryRequest(
        path=path,
        library_name=library_name,
        version=version,
        ingest_mode=ingest_mode,
        embedding_mode=embedding_mode,
    )


def _print_response(response) -> None:
    print(
        json.dumps(
            {
                "graph_id": response.graph_id,
                "library_name": response.library_name,
                "version": response.version,
                "counts": response.counts,
                "embedding_summary": {
                    "enabled": response.embedding_summary.enabled,
                    "embedded_nodes": response.embedding_summary.embedded_nodes,
                    "provider": response.embedding_summary.provider,
                    "model": response.embedding_summary.model,
                },
                "duration_ms": response.duration_ms,
                "warnings": response.warnings,
                "executed_stages": response.executed_stages,
            },
            indent=2,
        )
    )


def _print_search_response(response) -> None:
    print(
        json.dumps(
            {
                "query": response.query,
                "graph_scope": {
                    "requested_graph_id": response.graph_scope.requested_graph_id,
                    "requested_graph_ids": response.graph_scope.requested_graph_ids,
                    "resolved_graph_ids": response.graph_scope.resolved_graph_ids,
                },
                "routed_graphs": [{"graph_id": item.graph_id, "score": item.score} for item in response.routed_graphs],
                "candidate_count": response.candidate_count,
                "top_matches": [
                    {
                        "node_id": item.node_id,
                        "graph_id": item.graph_id,
                        "kind": item.kind,
                        "name": item.name,
                        "qualified_name": item.qualified_name,
                        "summary": item.summary,
                        "score": item.score,
                    }
                    for item in response.top_matches
                ],
                "context_preview": response.context_preview,
                "warnings": response.warnings,
            },
            indent=2,
        )
    )


def _select_library_path(scan_root: Path) -> Path | None:
    current_root = _resolve_library_scan_root(scan_root)
    candidates = _discover_library_candidates(current_root)
    print(f"\nLibrary search root: {current_root}")
    if not candidates:
        return _prompt_manual_path()
    selected_name = _prompt_library_search(candidates)
    if selected_name is None:
        return None
    if selected_name == "[Enter manual path]":
        return _prompt_manual_path()
    return next((path for path in candidates if path.name == selected_name), None)


def _discover_library_candidates(scan_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for child in sorted(scan_root.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if _looks_like_python_library(child):
            candidates.append(child)
    return candidates


def _looks_like_python_library(path: Path) -> bool:
    package_dir = path / path.name.replace("-", "_")
    if (path / "pyproject.toml").exists():
        return True
    if (path / "setup.py").exists():
        return True
    if (path / "src").is_dir():
        return True
    if (package_dir / "__init__.py").exists():
        return True
    python_files = list(path.glob("*.py"))
    return bool(python_files)


def _resolve_library_scan_root(scan_root: Path) -> Path:
    if scan_root.name == "site-packages":
        return scan_root
    unix_candidates = sorted(scan_root.glob("lib/python*/site-packages"))
    if unix_candidates:
        return unix_candidates[0]
    windows_candidate = scan_root / "Lib" / "site-packages"
    if windows_candidate.is_dir():
        return windows_candidate
    return scan_root


def _default_library_scan_root() -> Path:
    virtual_env = os.getenv("VIRTUAL_ENV")
    if virtual_env:
        env_root = Path(virtual_env).expanduser().resolve()
        if env_root.exists():
            return _resolve_library_scan_root(env_root)
    sys_prefix = Path(sys.prefix).expanduser().resolve()
    if sys_prefix.exists():
        return _resolve_library_scan_root(sys_prefix)
    return _resolve_library_scan_root(Path.cwd())


def _filter_library_candidates(candidates: list[Path], query: str) -> list[Path]:
    normalized_query = query.strip().lower()
    if not normalized_query:
        return candidates[:20]
    prefix_matches = [path for path in candidates if path.name.lower().startswith(normalized_query)]
    substring_matches = [
        path
        for path in candidates
        if normalized_query in path.name.lower() and path not in prefix_matches
    ]
    return (prefix_matches + substring_matches)[:20]


def _print_library_matches(matches: list[Path]) -> None:
    for index, path in enumerate(matches, start=1):
        print(f"{index}. {path.name}  [{path}]")


def _prompt_optional(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _prompt_manual_path() -> Path | None:
    manual = input("Enter absolute or relative path: ").strip()
    if not manual:
        return None
    selected = Path(manual).expanduser()
    if not selected.is_absolute():
        selected = (Path.cwd() / selected).resolve()
    return selected if selected.exists() and selected.is_dir() else None


def _prompt_library_search(candidates: list[Path]) -> str | None:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import FuzzyWordCompleter
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style

    names = [path.name for path in candidates]
    completer = FuzzyWordCompleter(["[Enter manual path]", *names], WORD=True)
    style = Style.from_dict(
        {
            "prompt": "#87ceeb bold",
            "label": "#bbbbbb bold",
            "hint": "#ffaa00 bold",
        }
    )
    while True:
        value = prompt(
            HTML("<prompt>?</prompt> <label>Search or select a library to ingest: </label>"),
            completer=completer,
            complete_while_typing=True,
            style=style,
        ).strip()
        if not value:
            continue
        lowered = value.lower()
        if lowered in {"q", "quit", "exit"}:
            return None
        if value == "[Enter manual path]":
            return value
        exact = next((name for name in names if name.lower() == lowered), None)
        if exact is not None:
            return exact
        filtered = _filter_library_candidates(candidates, value)
        if len(filtered) == 1:
            return filtered[0].name
        if filtered:
            print("Suggestions:")
            _print_library_matches(filtered[:10])
            continue
        print("No packages matched. Choose a suggestion or type '[Enter manual path]'.")


if __name__ == "__main__":
    raise SystemExit(main())
