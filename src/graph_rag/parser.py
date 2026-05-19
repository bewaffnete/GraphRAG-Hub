from __future__ import annotations

import ast
from pathlib import Path

from .ast_enrichment import (
    compute_complexity_metrics,
    extract_constants,
    extract_internal_calls,
    extract_logic_skeleton,
)
from .docstrings import parse_docstring
from .metadata import detect_library_metadata
from .models import (
    ClassInfo,
    CodeExample,
    FunctionInfo,
    ImportInfo,
    LibrarySnapshot,
    ModuleInfo,
    ParameterInfo,
    ReturnInfo,
)


class PythonLibraryParser:
    """
    A parser for extracting structured metadata from a Python library.

    It traverses the filesystem, parses Python files into ASTs, and 
    extracts information about modules, classes, functions, imports, 
    and docstrings.
    """

    def __init__(self, root: str | Path):
        """
        Initialize the parser with a library root directory.

        Args:
            root (str | Path): Path to the library's root directory.
        """
        self.root = Path(root).resolve()
        self.source_roots = self._detect_source_roots()
        self.package_prefix = self._detect_package_prefix()

    def parse(self) -> LibrarySnapshot:
        """
        Perform a full parse of the library.

        Returns:
            LibrarySnapshot: A structured snapshot of the entire library.
        """
        metadata = detect_library_metadata(self.root)
        snapshot = LibrarySnapshot(metadata=metadata)

        if metadata.readme_path:
            readme_examples = extract_examples_from_readme(Path(metadata.readme_path))
        else:
            readme_examples = []

        for path in sorted(self.root.rglob("*.py")):
            if self._should_skip(path):
                continue
            module = self._parse_module(path)
            if readme_examples and not snapshot.modules:
                module.examples.extend(readme_examples)
            snapshot.modules.append(module)

        return snapshot

    def _should_skip(self, path: Path) -> bool:
        """
        Determine if a given file path should be skipped during parsing.

        Args:
            path (Path): File path to check.

        Returns:
            bool: True if the file should be ignored, False otherwise.
        """
        try:
            relative_parts = path.relative_to(self.root).parts
        except ValueError:
            relative_parts = path.parts
        skipped_parts = {
            ".git",
            "__pycache__",
            "build",
            "dist",
            ".mypy_cache",
            ".pytest_cache",
            "tests",
            "testing",
            "benchmarks",
            "docs",
        }
        return any(part in relative_parts for part in skipped_parts)

    def _is_public(self, name: str) -> bool:
        """
        Check if a name (class, function, etc.) is considered public.

        Args:
            name (str): The name to check.

        Returns:
            bool: True if public or a double-underscore name, False if it 
                  starts with a single underscore.
        """
        if not name:
            return True
        last_part = name.split(".")[-1]
        if last_part.startswith("_"):
            return last_part.startswith("__") and last_part.endswith("__")
        return True

    def _parse_module(self, path: Path) -> ModuleInfo:
        """
        Parse a single Python module file.

        Args:
            path (Path): Path to the .py file.

        Returns:
            ModuleInfo: Metadata extracted from the module.
        """
        source = self._read_source(path)
        tree = ast.parse(source, filename=str(path))
        relative_path = path.relative_to(self.root)
        module_name = self._module_name_for_path(path)

        module_docstring = ast.get_docstring(tree)
        _, _, _, _, module_examples = parse_docstring(module_docstring)
        module = ModuleInfo(
            name=module_name,
            path=str(relative_path),
            docstring=module_docstring,
            imports=self._extract_imports(tree),
            examples=module_examples,
        )

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if self._is_public(node.name):
                    module.classes.append(self._parse_class(node, module_name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_public(node.name):
                    module.functions.append(self._parse_function(node, module_name, class_name=None))

        return module

    def _detect_source_roots(self) -> list[Path]:
        """
        Identify likely source root directories (e.g., 'src' or root).

        Returns:
            list[Path]: List of valid source root paths.
        """
        candidates = [self.root / "src", self.root]
        return [candidate for candidate in candidates if candidate.exists()]

    def _module_name_for_path(self, path: Path) -> str:
        """
        Calculate the dot-separated module name for a file path.

        Args:
            path (Path): Path to the Python file.

        Returns:
            str: The computed module name.
        """
        for source_root in self.source_roots:
            try:
                relative = path.relative_to(source_root)
                module_name = ".".join(relative.with_suffix("").parts)
                if self.package_prefix and source_root == self.root:
                    if module_name == "__init__":
                        return f"{self.package_prefix}.__init__"
                    return f"{self.package_prefix}.{module_name}"
                return module_name
            except ValueError:
                continue
        return ".".join(path.relative_to(self.root).with_suffix("").parts)

    def _detect_package_prefix(self) -> str | None:
        """
        Detect if the library root itself is a package (contains __init__.py).

        Returns:
            str | None: The package name if detected, else None.
        """
        if (self.root / "__init__.py").exists():
            return self.root.name
        return None

    def _extract_imports(self, tree: ast.Module) -> list[ImportInfo]:
        """
        Extract all import statements from an AST module.

        Args:
            tree (ast.Module): The AST of the module.

        Returns:
            list[ImportInfo]: List of extracted import metadata.
        """
        imports: list[ImportInfo] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.append(
                    ImportInfo(
                        module=None,
                        names=[alias.name for alias in node.names],
                        lineno=node.lineno,
                    )
                )
            elif isinstance(node, ast.ImportFrom):
                imports.append(
                    ImportInfo(
                        module=node.module,
                        names=[alias.name for alias in node.names],
                        lineno=node.lineno,
                    )
                )
        return imports

    def _parse_class(self, node: ast.ClassDef, module_name: str) -> ClassInfo:
        """
        Parse a class definition node.

        Args:
            node (ast.ClassDef): The class definition AST node.
            module_name (str): Name of the containing module.

        Returns:
            ClassInfo: Extracted class metadata.
        """
        docstring = ast.get_docstring(node)
        _, _, _, _, examples = parse_docstring(docstring)
        class_info = ClassInfo(
            name=node.name,
            qualname=f"{module_name}.{node.name}",
            module=module_name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            docstring=docstring,
            bases=[self._expr_to_str(base) for base in node.bases],
            decorators=[self._expr_to_str(decorator) for decorator in node.decorator_list],
            examples=examples,
            internal_calls=extract_internal_calls(node),
            constants_used=extract_constants(node),
            complexity_metrics=compute_complexity_metrics(node).to_dict(),
        )
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_public(child.name):
                    class_info.methods.append(self._parse_function(child, module_name, class_name=node.name))
        return class_info

    def _parse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module_name: str,
        class_name: str | None,
    ) -> FunctionInfo:
        """
        Parse a function or method definition node.

        Args:
            node (ast.FunctionDef | ast.AsyncFunctionDef): The function AST node.
            module_name (str): Name of the containing module.
            class_name (str | None): Name of the parent class, if any.

        Returns:
            FunctionInfo: Extracted function metadata.
        """
        docstring = ast.get_docstring(node)
        _, doc_params, doc_returns, doc_raises, examples = parse_docstring(docstring)
        parameters = self._build_parameters(node.args, doc_params)
        returns = self._merge_return_info(node, doc_returns)
        qualname = f"{module_name}.{node.name}" if class_name is None else f"{module_name}.{class_name}.{node.name}"
        return FunctionInfo(
            name=node.name,
            qualname=qualname,
            module=module_name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            signature=self._build_signature(node),
            docstring=docstring,
            decorators=[self._expr_to_str(decorator) for decorator in node.decorator_list],
            parameters=parameters,
            returns=returns,
            raises=doc_raises,
            examples=examples,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=class_name is not None,
            logic_skeleton=extract_logic_skeleton(node),
            internal_calls=extract_internal_calls(node),
            constants_used=extract_constants(node),
            complexity_metrics=compute_complexity_metrics(node).to_dict(),
        )

    def _build_parameters(self, args: ast.arguments, doc_params: list[ParameterInfo]) -> list[ParameterInfo]:
        """
        Construct ParameterInfo objects by merging AST data with docstring info.

        Args:
            args (ast.arguments): The arguments node from the AST.
            doc_params (list[ParameterInfo]): Parameter info from the docstring.

        Returns:
            list[ParameterInfo]: Merged parameter metadata.
        """
        doc_params_by_name = {param.name: param for param in doc_params}
        positional = list(args.posonlyargs) + list(args.args)
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        parameters: list[ParameterInfo] = []

        for arg, default in zip(positional, defaults):
            doc_param = doc_params_by_name.get(arg.arg)
            parameters.append(
                ParameterInfo(
                    name=arg.arg,
                    annotation=self._expr_to_str(arg.annotation),
                    default=self._expr_to_str(default),
                    kind="positional",
                    description=doc_param.description if doc_param else None,
                )
            )

        if args.vararg:
            doc_param = doc_params_by_name.get(args.vararg.arg)
            parameters.append(
                ParameterInfo(
                    name=args.vararg.arg,
                    annotation=self._expr_to_str(args.vararg.annotation),
                    default=None,
                    kind="vararg",
                    description=doc_param.description if doc_param else None,
                )
            )

        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            doc_param = doc_params_by_name.get(arg.arg)
            parameters.append(
                ParameterInfo(
                    name=arg.arg,
                    annotation=self._expr_to_str(arg.annotation),
                    default=self._expr_to_str(default),
                    kind="kwonly",
                    description=doc_param.description if doc_param else None,
                )
            )

        if args.kwarg:
            doc_param = doc_params_by_name.get(args.kwarg.arg)
            parameters.append(
                ParameterInfo(
                    name=args.kwarg.arg,
                    annotation=self._expr_to_str(args.kwarg.annotation),
                    default=None,
                    kind="kwarg",
                    description=doc_param.description if doc_param else None,
                )
            )

        return parameters

    def _merge_return_info(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        doc_returns: ReturnInfo | None,
    ) -> ReturnInfo | None:
        """
        Combine AST return type annotations with docstring return descriptions.

        Args:
            node (ast.FunctionDef | ast.AsyncFunctionDef): The function AST node.
            doc_returns (ReturnInfo | None): Return info from the docstring.

        Returns:
            ReturnInfo | None: Merged return metadata.
        """
        annotation = self._expr_to_str(node.returns)
        if annotation or doc_returns:
            return ReturnInfo(
                annotation=annotation or (doc_returns.annotation if doc_returns else None),
                description=doc_returns.description if doc_returns else None,
            )
        return None

    def _build_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """
        Reconstruct a string representation of the function's signature.

        Args:
            node (ast.FunctionDef | ast.AsyncFunctionDef): The function AST node.

        Returns:
            str: The reconstructed signature (e.g., 'def func(a: int) -> bool').
        """
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args = []
        for arg in node.args.posonlyargs:
            args.append(self._format_arg(arg))
        if node.args.posonlyargs:
            args.append("/")
        for arg in node.args.args:
            args.append(self._format_arg(arg))
        if node.args.vararg:
            args.append(f"*{self._format_arg(node.args.vararg)}")
        elif node.args.kwonlyargs:
            args.append("*")
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            args.append(self._format_arg(arg, default))
        if node.args.kwarg:
            args.append(f"**{self._format_arg(node.args.kwarg)}")
        signature = f"{prefix} {node.name}({', '.join(args)})"
        returns = self._expr_to_str(node.returns)
        if returns:
            signature += f" -> {returns}"
        return signature

    def _format_arg(self, arg: ast.arg, default: ast.AST | None = None) -> str:
        """Format an individual argument with its annotation and default value."""
        result = arg.arg
        annotation = self._expr_to_str(arg.annotation)
        if annotation:
            result += f": {annotation}"
        default_str = self._expr_to_str(default)
        if default_str:
            result += f" = {default_str}"
        return result

    def _expr_to_str(self, node: ast.AST | None) -> str | None:
        """Unparse an AST expression node back into source code."""
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return None

    def _read_source(self, path: Path) -> str:
        """Read source code from a file, handling basic encoding issues."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")


def extract_examples_from_readme(path: Path) -> list[CodeExample]:
    """
    Extract Python code blocks from a README markdown file.

    Args:
        path (Path): Path to the README file.

    Returns:
        list[CodeExample]: List of extracted code examples.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    examples: list[CodeExample] = []
    for block in text.split("```"):
        snippet = block.strip()
        if not snippet or "\n" not in snippet:
            continue
        first_line, rest = snippet.split("\n", 1)
        language = first_line.strip().lower()
        if language and language not in {"python", "py"}:
            continue
        code = rest.strip() if language else snippet.strip()
        if code:
            examples.append(CodeExample(code=code, source="readme"))
    return examples


def parse_python_library(root: str | Path) -> LibrarySnapshot:
    """
    Entry point to parse a Python library and return a snapshot.

    Args:
        root (str | Path): Path to the library's root directory.

    Returns:
        LibrarySnapshot: The generated library snapshot.
    """
    return PythonLibraryParser(root).parse()
