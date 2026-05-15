from __future__ import annotations

import re
from collections.abc import Iterable

from .models import CodeExample, ParameterInfo, RaiseInfo, ReturnInfo


SECTION_RE = re.compile(r"^(Args|Arguments|Parameters|Returns|Yields|Raises)\s*:\s*$")
PARAM_RE = re.compile(r"^\s{0,4}([*\w][\w\d_]*)(?:\s*\(([^)]+)\))?\s*:\s*(.+)$")
RAISE_RE = re.compile(r"^\s{0,4}([\w.]+)\s*:\s*(.+)$")
CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
PROMPT_BLOCK_RE = re.compile(r"(^>>> .+(?:\n(?:\.\.\.|\S).+)*)", re.MULTILINE)


def _split_sections(docstring: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"summary": []}
    current = "summary"
    for line in docstring.splitlines():
        match = SECTION_RE.match(line.strip())
        if match:
            current = match.group(1).lower()
            sections[current] = []
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _cleanup_description(lines: Iterable[str]) -> str | None:
    text = "\n".join(line.rstrip() for line in lines).strip()
    return text or None


def parse_docstring(docstring: str | None) -> tuple[dict[str, str], list[ParameterInfo], ReturnInfo | None, list[RaiseInfo], list[CodeExample]]:
    if not docstring:
        return {}, [], None, [], []

    sections = _split_sections(docstring)
    parsed_sections = {
        name: _cleanup_description(lines) or ""
        for name, lines in sections.items()
        if name != "summary"
    }

    parameters: list[ParameterInfo] = []
    for raw_line in sections.get("args", []) + sections.get("arguments", []) + sections.get("parameters", []):
        match = PARAM_RE.match(raw_line)
        if not match:
            continue
        parameters.append(
            ParameterInfo(
                name=match.group(1),
                annotation=match.group(2),
                description=match.group(3).strip(),
                kind="docstring",
            )
        )

    returns = None
    returns_block = sections.get("returns", []) or sections.get("yields", [])
    returns_text = _cleanup_description(returns_block)
    if returns_text:
        first_line, *rest = returns_text.splitlines()
        annotation = None
        description = returns_text
        if ":" in first_line:
            possible_type, possible_desc = first_line.split(":", 1)
            if possible_type.strip() and " " not in possible_type.strip():
                annotation = possible_type.strip()
                description = "\n".join([possible_desc.strip(), *rest]).strip()
        returns = ReturnInfo(annotation=annotation, description=description or None)

    raises: list[RaiseInfo] = []
    for raw_line in sections.get("raises", []):
        match = RAISE_RE.match(raw_line)
        if not match:
            continue
        raises.append(RaiseInfo(exception=match.group(1), description=match.group(2).strip()))

    examples: list[CodeExample] = []
    for block in CODE_BLOCK_RE.findall(docstring):
        snippet = block.strip()
        if snippet:
            examples.append(CodeExample(code=snippet, source="docstring"))
    for block in PROMPT_BLOCK_RE.findall(docstring):
        snippet = block.strip()
        if snippet:
            examples.append(CodeExample(code=snippet, source="docstring"))

    return parsed_sections, parameters, returns, raises, examples
