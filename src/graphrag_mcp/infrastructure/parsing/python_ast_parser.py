"""Python AST parser that emits public-API graph entities."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from graphrag_mcp.application.ports.parser_port import ParsedGraph, ParserPort
from graphrag_mcp.domain.entities.graph_edge import GraphEdge
from graphrag_mcp.domain.entities.graph_node import GraphNode
from graphrag_mcp.domain.exceptions import ParseFailureError
from graphrag_mcp.domain.services.graph_identity_policy import GraphIdentityPolicy
from graphrag_mcp.infrastructure.parsing.docstring_parser import annotation_to_text, extract_examples, summarize_docstring
from graphrag_mcp.infrastructure.parsing.metadata_extractor import extract_module_all, is_public_name, module_is_public


@dataclass(slots=True)
class _ModuleContext:
    module_name: str
    source_path: str
    source_text: str
    exports: set[str]


class PythonAstParser(ParserPort):
    def __init__(self, identity_policy: GraphIdentityPolicy | None = None) -> None:
        self._identity_policy = identity_policy or GraphIdentityPolicy()

    def parse_public_api(
        self,
        *,
        root_path: Path,
        library_name: str,
        version: str,
        graph_id: str,
    ) -> ParsedGraph:
        package_root = _resolve_package_root(root_path, library_name)
        parsed_graph = ParsedGraph(
            library_name=library_name,
            version=version,
            root_path=root_path,
        )
        parsed_graph.nodes.append(
            GraphNode(
                node_id=self._identity_policy.build_node_id(graph_id, "Library", graph_id),
                graph_id=graph_id,
                kind="Library",
                name=library_name,
                display_name=library_name,
                qualified_name=graph_id,
                source_path="",
                line_start=None,
                line_end=None,
                summary=f"Public API graph for {library_name} {version}.",
                is_public=True,
                metadata={
                    "version": version,
                    "status": "building",
                    "root_path": str(root_path),
                },
            )
        )
        for file_path in sorted(package_root.rglob("*.py")):
            source_path = file_path.relative_to(root_path).as_posix()
            module_name = _module_name_for_file(root_path=root_path, file_path=file_path)
            if not module_is_public(module_name):
                continue
            try:
                source_text = file_path.read_text(encoding="utf-8")
                module_ast = ast.parse(source_text, filename=str(file_path))
            except SyntaxError as exc:
                parsed_graph.warnings.append(f"Skipped {source_path} due to syntax error: {exc.msg}")
                continue
            except OSError as exc:
                raise ParseFailureError(
                    code="PARSE_FAILED",
                    message=f"Could not read source file {file_path}: {exc}",
                    hint="Verify that the library files are readable.",
                ) from exc

            context = _ModuleContext(
                module_name=module_name,
                source_path=source_path,
                source_text=source_text,
                exports=extract_module_all(module_ast),
            )
            self._parse_module(context=context, module_ast=module_ast, graph_id=graph_id, parsed_graph=parsed_graph)
        return parsed_graph

    def _parse_module(self, *, context: _ModuleContext, module_ast: ast.Module, graph_id: str, parsed_graph: ParsedGraph) -> None:
        module_id = self._identity_policy.build_node_id(graph_id, "Module", context.module_name)
        library_id = self._identity_policy.build_node_id(graph_id, "Library", graph_id)
        module_docstring = ast.get_docstring(module_ast)
        module_node = GraphNode(
            node_id=module_id,
            graph_id=graph_id,
            kind="Module",
            name=context.module_name.rsplit(".", 1)[-1],
            display_name=context.module_name,
            qualified_name=context.module_name,
            source_path=context.source_path,
            line_start=1,
            line_end=len(context.source_text.splitlines()) or 1,
            summary=summarize_docstring(module_docstring, f"Public module {context.module_name}."),
            docstring=module_docstring,
            is_public=True,
            metadata={"module_path": context.module_name},
        )
        parsed_graph.nodes.append(module_node)
        parsed_graph.edges.append(GraphEdge(source_node_id=library_id, target_node_id=module_id, type="CONTAINS"))

        for stmt in module_ast.body:
            if isinstance(stmt, ast.ClassDef):
                self._parse_class(stmt=stmt, context=context, graph_id=graph_id, module_id=module_id, parsed_graph=parsed_graph)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._parse_function(
                    stmt=stmt,
                    context=context,
                    graph_id=graph_id,
                    parent_id=module_id,
                    parent_kind="Module",
                    class_name=None,
                    parsed_graph=parsed_graph,
                )

    def _parse_class(self, *, stmt: ast.ClassDef, context: _ModuleContext, graph_id: str, module_id: str, parsed_graph: ParsedGraph) -> None:
        if not _is_exported(stmt.name, context.exports):
            return
        qualified_name = f"{context.module_name}.{stmt.name}"
        class_id = self._identity_policy.build_node_id(graph_id, "Class", qualified_name)
        docstring = ast.get_docstring(stmt)
        parsed_graph.nodes.append(
            GraphNode(
                node_id=class_id,
                graph_id=graph_id,
                kind="Class",
                name=stmt.name,
                display_name=stmt.name,
                qualified_name=qualified_name,
                source_path=context.source_path,
                line_start=getattr(stmt, "lineno", None),
                line_end=getattr(stmt, "end_lineno", None),
                summary=summarize_docstring(docstring, f"Public class {stmt.name}."),
                docstring=docstring,
                logic_skeleton=_build_logic_skeleton(stmt.body),
                source_excerpt=_excerpt(context.source_text, stmt),
                is_public=True,
                metadata={"bases": [annotation_to_text(base) for base in stmt.bases if annotation_to_text(base)]},
            )
        )
        parsed_graph.edges.append(GraphEdge(source_node_id=module_id, target_node_id=class_id, type="DECLARES"))

        for child in stmt.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._parse_function(
                    stmt=child,
                    context=context,
                    graph_id=graph_id,
                    parent_id=class_id,
                    parent_kind="Class",
                    class_name=stmt.name,
                    parsed_graph=parsed_graph,
                )

    def _parse_function(
        self,
        *,
        stmt: ast.FunctionDef | ast.AsyncFunctionDef,
        context: _ModuleContext,
        graph_id: str,
        parent_id: str,
        parent_kind: str,
        class_name: str | None,
        parsed_graph: ParsedGraph,
    ) -> None:
        if not _is_exported(stmt.name, context.exports if class_name is None else set()):
            return
        qualified_name = f"{context.module_name}.{stmt.name}" if class_name is None else f"{context.module_name}.{class_name}.{stmt.name}"
        function_id = self._identity_policy.build_node_id(graph_id, "Function", qualified_name)
        signature = _signature_for_function(stmt)
        docstring = ast.get_docstring(stmt)
        parsed_graph.nodes.append(
            GraphNode(
                node_id=function_id,
                graph_id=graph_id,
                kind="Function",
                name=stmt.name,
                display_name=stmt.name,
                qualified_name=qualified_name,
                source_path=context.source_path,
                line_start=getattr(stmt, "lineno", None),
                line_end=getattr(stmt, "end_lineno", None),
                summary=summarize_docstring(docstring, f"Public function {stmt.name}."),
                docstring=docstring,
                signature=signature,
                logic_skeleton=_build_logic_skeleton(stmt.body),
                source_excerpt=_excerpt(context.source_text, stmt),
                is_public=True,
                metadata={
                    "is_async": isinstance(stmt, ast.AsyncFunctionDef),
                    "is_method": class_name is not None,
                    "is_classmethod": any(_decorator_name(deco) == "classmethod" for deco in stmt.decorator_list),
                    "is_staticmethod": any(_decorator_name(deco) == "staticmethod" for deco in stmt.decorator_list),
                    "returns_annotation": annotation_to_text(stmt.returns),
                    "raises": _extract_raises(stmt),
                },
            )
        )
        parsed_graph.edges.append(
            GraphEdge(
                source_node_id=parent_id,
                target_node_id=function_id,
                type="HAS_METHOD" if parent_kind == "Class" else "DECLARES",
            )
        )
        self._add_parameters(stmt=stmt, context=context, graph_id=graph_id, function_id=function_id, parsed_graph=parsed_graph)
        self._add_return(stmt=stmt, context=context, graph_id=graph_id, function_id=function_id, parsed_graph=parsed_graph)
        self._add_exceptions(stmt=stmt, context=context, graph_id=graph_id, function_id=function_id, parsed_graph=parsed_graph)
        self._add_examples(
            docstring=docstring,
            qualified_name=qualified_name,
            context=context,
            graph_id=graph_id,
            function_id=function_id,
            parsed_graph=parsed_graph,
        )

    def _add_parameters(self, *, stmt, context: _ModuleContext, graph_id: str, function_id: str, parsed_graph: ParsedGraph) -> None:
        for position, arg in enumerate(stmt.args.posonlyargs + stmt.args.args + stmt.args.kwonlyargs):
            if arg.arg in {"self", "cls"}:
                continue
            qualified_name = f"{function_id.rsplit(':', 1)[-1]}:{arg.arg}"
            parameter_id = self._identity_policy.build_node_id(graph_id, "Parameter", qualified_name)
            parsed_graph.nodes.append(
                GraphNode(
                    node_id=parameter_id,
                    graph_id=graph_id,
                    kind="Parameter",
                    name=arg.arg,
                    display_name=arg.arg,
                    qualified_name=qualified_name,
                    source_path=context.source_path,
                    line_start=getattr(arg, "lineno", getattr(stmt, "lineno", None)),
                    line_end=getattr(arg, "end_lineno", getattr(stmt, "lineno", None)),
                    summary=f"Public API parameter {arg.arg}.",
                    is_public=True,
                    metadata={
                        "annotation": annotation_to_text(arg.annotation),
                        "position": position,
                        "parameter_kind": "keyword_only" if arg in stmt.args.kwonlyargs else "positional",
                    },
                )
            )
            parsed_graph.edges.append(GraphEdge(source_node_id=function_id, target_node_id=parameter_id, type="HAS_PARAM"))

    def _add_return(self, *, stmt, context: _ModuleContext, graph_id: str, function_id: str, parsed_graph: ParsedGraph) -> None:
        annotation = annotation_to_text(stmt.returns)
        if annotation is None:
            return
        qualified_name = f"{function_id.rsplit(':', 1)[-1]}:return"
        return_id = self._identity_policy.build_node_id(graph_id, "ReturnType", qualified_name)
        parsed_graph.nodes.append(
            GraphNode(
                node_id=return_id,
                graph_id=graph_id,
                kind="ReturnType",
                name="return",
                display_name="return",
                qualified_name=qualified_name,
                source_path=context.source_path,
                line_start=getattr(stmt, "lineno", None),
                line_end=getattr(stmt, "end_lineno", None),
                summary=f"Declared return type {annotation}.",
                is_public=True,
                metadata={"annotation": annotation},
            )
        )
        parsed_graph.edges.append(GraphEdge(source_node_id=function_id, target_node_id=return_id, type="RETURNS"))

    def _add_exceptions(self, *, stmt, context: _ModuleContext, graph_id: str, function_id: str, parsed_graph: ParsedGraph) -> None:
        for name in _extract_raises(stmt):
            qualified_name = f"{function_id.rsplit(':', 1)[-1]}:{name}"
            exception_id = self._identity_policy.build_node_id(graph_id, "Exception", qualified_name)
            parsed_graph.nodes.append(
                GraphNode(
                    node_id=exception_id,
                    graph_id=graph_id,
                    kind="Exception",
                    name=name,
                    display_name=name,
                    qualified_name=qualified_name,
                    source_path=context.source_path,
                    line_start=getattr(stmt, "lineno", None),
                    line_end=getattr(stmt, "end_lineno", None),
                    summary=f"Public API may raise {name}.",
                    is_public=True,
                    metadata={"origin": "raise_statement"},
                )
            )
            parsed_graph.edges.append(GraphEdge(source_node_id=function_id, target_node_id=exception_id, type="RAISES"))

    def _add_examples(
        self,
        *,
        docstring: str | None,
        qualified_name: str,
        context: _ModuleContext,
        graph_id: str,
        function_id: str,
        parsed_graph: ParsedGraph,
    ) -> None:
        for index, code in enumerate(extract_examples(docstring)):
            example_qualified_name = f"{qualified_name}:example:{index}"
            example_id = self._identity_policy.build_node_id(graph_id, "Example", example_qualified_name)
            parsed_graph.nodes.append(
                GraphNode(
                    node_id=example_id,
                    graph_id=graph_id,
                    kind="Example",
                    name=f"example_{index}",
                    display_name=f"example_{index}",
                    qualified_name=example_qualified_name,
                    source_path=context.source_path,
                    line_start=None,
                    line_end=None,
                    summary=f"Usage example for {qualified_name}.",
                    source_excerpt=code,
                    is_public=True,
                    metadata={"example_source": "docstring", "code": code},
                )
            )
            parsed_graph.edges.append(GraphEdge(source_node_id=function_id, target_node_id=example_id, type="USES_EXAMPLE"))


def _resolve_package_root(root_path: Path, library_name: str) -> Path:
    candidate = root_path / library_name
    if candidate.exists() and candidate.is_dir():
        return candidate
    return root_path


def _module_name_for_file(*, root_path: Path, file_path: Path) -> str:
    relative = file_path.relative_to(root_path).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_exported(name: str, exports: set[str]) -> bool:
    return name in exports if exports else is_public_name(name)


def _signature_for_function(stmt: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        return f"{stmt.name}{ast.unparse(stmt.args)}"
    except Exception:
        return f"{stmt.name}(...)"


def _decorator_name(node: ast.AST) -> str | None:
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _extract_raises(stmt: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    names: list[str] = []
    for child in ast.walk(stmt):
        if isinstance(child, ast.Raise) and child.exc is not None:
            if isinstance(child.exc, ast.Call):
                text = annotation_to_text(child.exc.func)
            else:
                text = annotation_to_text(child.exc)
            if text:
                names.append(text)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _build_logic_skeleton(body: list[ast.stmt]) -> str | None:
    labels: list[str] = []
    for node in body:
        if isinstance(node, ast.Assign):
            labels.append("assign")
        elif isinstance(node, ast.AnnAssign):
            labels.append("annotate")
        elif isinstance(node, ast.If):
            labels.append("branch")
        elif isinstance(node, ast.For):
            labels.append("loop")
        elif isinstance(node, ast.While):
            labels.append("while")
        elif isinstance(node, ast.Try):
            labels.append("try")
        elif isinstance(node, ast.Return):
            labels.append("return")
        elif isinstance(node, ast.Expr):
            labels.append("expr")
    if not labels:
        return None
    return " -> ".join(labels[:8])


def _excerpt(source_text: str, node: ast.AST, max_lines: int = 12) -> str | None:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    if lineno is None or end_lineno is None:
        return None
    lines = source_text.splitlines()
    excerpt = lines[lineno - 1 : min(end_lineno, lineno - 1 + max_lines)]
    return "\n".join(excerpt) if excerpt else None
