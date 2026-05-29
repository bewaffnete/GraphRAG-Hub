"""Deterministic docstring helpers."""

import ast


def summarize_docstring(docstring: str | None, fallback: str) -> str:
    if not docstring:
        return fallback
    first_line = next((line.strip() for line in docstring.splitlines() if line.strip()), "")
    return first_line or fallback


def extract_examples(docstring: str | None) -> list[str]:
    if not docstring:
        return []
    lines = [line.rstrip() for line in docstring.splitlines()]
    examples: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(">>>") or (current and stripped.startswith("...")):
            current.append(stripped)
            continue
        if current:
            examples.append("\n".join(current))
            current = []
    if current:
        examples.append("\n".join(current))
    return examples


def annotation_to_text(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None
