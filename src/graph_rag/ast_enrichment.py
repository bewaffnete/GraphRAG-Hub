"""AST enrichment utilities for extracting structural code information.

Provides logic skeleton extraction, call graph mapping, constants collection,
and complexity metrics computation from Python AST nodes.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field


@dataclass
class ComplexityMetrics:
    """
    Container for code complexity and structural metrics.

    Attributes:
        cyclomatic_complexity (int): Number of linearly independent paths.
        max_nesting_depth (int): Maximum depth of nested control structures.
        branch_count (int): Total number of decision branches.
    """

    cyclomatic_complexity: int = 1
    max_nesting_depth: int = 0
    branch_count: int = 0

    def to_dict(self) -> dict:
        """Convert metrics to a dictionary."""
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "max_nesting_depth": self.max_nesting_depth,
            "branch_count": self.branch_count,
        }


# ---------------------------------------------------------------------------
# 1. Logic skeleton extraction
# ---------------------------------------------------------------------------


class _SkeletonTransformer(ast.NodeTransformer):
    """
    Strips function bodies to only keep control flow, calls, and returns.

    Keeps:
      - if / elif / else, for, while (full structure)
      - return, yield, yield from, raise
      - function/method calls (including assignments whose value is a call)
      - with statements (context managers)
    Removes:
      - simple variable assignments (unless the value is a call)
      - try/except wrappers (inlines try-body only)
      - standalone expressions that are not calls
      - pass / break / continue (noise for LLM understanding)
      - docstrings (already stored separately)
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = self._filter_body(node.body)
        node.decorator_list = []
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    # --- keep control flow ------------------------------------------------

    def visit_If(self, node: ast.If) -> ast.If | None:
        node.body = self._filter_body(node.body)
        node.orelse = self._filter_body(node.orelse)
        if not node.body and not node.orelse:
            return None
        return node

    def visit_For(self, node: ast.For) -> ast.For | None:
        node.body = self._filter_body(node.body)
        node.orelse = self._filter_body(node.orelse)
        if not node.body:
            return None
        return node

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> ast.While | None:
        node.body = self._filter_body(node.body)
        node.orelse = self._filter_body(node.orelse)
        if not node.body:
            return None
        return node

    def visit_With(self, node: ast.With) -> ast.With | None:
        node.body = self._filter_body(node.body)
        if not node.body:
            return None
        return node

    visit_AsyncWith = visit_With

    # --- inline try bodies ------------------------------------------------

    def visit_Try(self, node: ast.Try) -> list[ast.stmt]:
        return self._filter_body(node.body)

    def visit_TryStar(self, node: ast.AST) -> list[ast.stmt]:
        return self._filter_body(getattr(node, "body", []))

    # --- keep returns / yields / raises -----------------------------------

    def visit_Return(self, node: ast.Return) -> ast.Return:
        return node

    def visit_Yield(self, node: ast.Yield) -> ast.Yield:
        return node

    def visit_YieldFrom(self, node: ast.YieldFrom) -> ast.YieldFrom:
        return node

    def visit_Raise(self, node: ast.Raise) -> ast.Raise:
        return node

    # --- assignments: keep only if RHS is a call --------------------------

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        if self._contains_call(node.value):
            return node
        return None

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign | None:
        if self._contains_call(node.value):
            return node
        return None

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign | None:
        if node.value and self._contains_call(node.value):
            return node
        return None

    # --- expressions: keep only calls -------------------------------------

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        # Skip docstrings
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        if self._contains_call(node.value):
            return node
        return None

    # --- drop noise -------------------------------------------------------

    def visit_Pass(self, node: ast.Pass) -> None:
        return None

    def visit_Break(self, node: ast.Break) -> None:
        return None

    def visit_Continue(self, node: ast.Continue) -> None:
        return None

    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None

    def visit_Assert(self, node: ast.Assert) -> None:
        return None

    # --- helpers ----------------------------------------------------------

    def _filter_body(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for stmt in stmts:
            transformed = self.visit(stmt)
            if transformed is None:
                continue
            if isinstance(transformed, list):
                result.extend(transformed)
            else:
                result.append(transformed)
        return result

    @staticmethod
    def _contains_call(node: ast.AST | None) -> bool:
        if node is None:
            return False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                return True
        return False


def extract_logic_skeleton(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """
    Extract a cleaned logic skeleton from a function/method AST node.

    The skeleton preserves control flow (if, for, while, with), function calls,
    returns, yields, and raises while stripping assignments and docstrings.

    Args:
        node (ast.FunctionDef | ast.AsyncFunctionDef): The function AST node.

    Returns:
        str | None: A string representing the logic skeleton, or None if empty.
    """
    try:
        tree_copy = copy.deepcopy(node)
        transformer = _SkeletonTransformer()
        transformed = transformer.visit(tree_copy)
        if transformed is None:
            return None
        # Unparse only the body to avoid repeating the signature
        body_lines = []
        for stmt in transformed.body:
            try:
                body_lines.append(ast.unparse(stmt))
            except Exception:
                continue
        skeleton = "\n".join(body_lines)
        return skeleton.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. Internal calls extraction
# ---------------------------------------------------------------------------


class _CallCollector(ast.NodeVisitor):
    """AST visitor that collects names of all function and method calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = self._resolve_call_name(node.func)
        if name:
            self.calls.append(name)
        self.generic_visit(node)

    @staticmethod
    def _resolve_call_name(node: ast.AST) -> str | None:
        """Recursively resolve the full name of a call (e.g., 'obj.method')."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = [node.attr]
            current = node.value
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            parts.reverse()
            return ".".join(parts)
        return None


def extract_internal_calls(node: ast.AST) -> list[str]:
    """
    Extract a deduplicated, sorted list of all called function/method names.

    Args:
        node (ast.AST): The AST node to inspect.

    Returns:
        list[str]: List of unique call names.
    """
    collector = _CallCollector()
    collector.visit(node)
    # Deduplicate while preserving first-seen order
    seen: set[str] = set()
    unique: list[str] = []
    for name in collector.calls:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


# ---------------------------------------------------------------------------
# 3. Constants extraction
# ---------------------------------------------------------------------------


class _ConstantCollector(ast.NodeVisitor):
    """AST visitor that collects meaningful string and numeric literals."""

    def __init__(self) -> None:
        self.constants: list[str] = []

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            # Skip docstrings (handled by parent Expr check) and empty strings
            if node.value.strip():
                self.constants.append(node.value)
        elif isinstance(node.value, (int, float)):
            # Skip trivial values 0, 1, -1 and booleans
            if not isinstance(node.value, bool) and node.value not in (0, 1, -1, 0.0, 1.0, -1.0):
                self.constants.append(str(node.value))
        self.generic_visit(node)


def extract_constants(node: ast.AST) -> list[str]:
    """
    Extract a deduplicated list of meaningful string and numeric constants.

    Args:
        node (ast.AST): The AST node to inspect.

    Returns:
        list[str]: List of unique constant values as strings.
    """
    collector = _ConstantCollector()
    collector.visit(node)
    seen: set[str] = set()
    unique: list[str] = []
    for value in collector.constants:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


# ---------------------------------------------------------------------------
# 4. Complexity metrics
# ---------------------------------------------------------------------------


class _ComplexityVisitor(ast.NodeVisitor):
    """AST visitor that computes complexity metrics based on control flow nodes."""

    def __init__(self) -> None:
        self.complexity = 1  # Base complexity
        self.branch_count = 0
        self._current_depth = 0
        self._max_depth = 0

    def _enter_branch(self) -> None:
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)

    def _exit_branch(self) -> None:
        self._current_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.branch_count += 1
        self._enter_branch()
        self.generic_visit(node)
        self._exit_branch()

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.branch_count += 1
        self._enter_branch()
        self.generic_visit(node)
        self._exit_branch()

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.branch_count += 1
        self._enter_branch()
        self.generic_visit(node)
        self._exit_branch()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.branch_count += 1
        self._enter_branch()
        self.generic_visit(node)
        self._exit_branch()

    def visit_With(self, node: ast.With) -> None:
        self._enter_branch()
        self.generic_visit(node)
        self._exit_branch()

    visit_AsyncWith = visit_With

    # Boolean operators add decision points
    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Each `and` / `or` adds (n_values - 1) decision points
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    # Comprehensions with if-clauses
    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.complexity += len(node.ifs)
        self.branch_count += len(node.ifs)
        self.generic_visit(node)


def compute_complexity_metrics(node: ast.AST) -> ComplexityMetrics:
    """
    Compute cyclomatic complexity, nesting depth, and branch count for an AST.

    Args:
        node (ast.AST): The AST node to analyze.

    Returns:
        ComplexityMetrics: The computed metrics.
    """
    visitor = _ComplexityVisitor()
    visitor.visit(node)
    return ComplexityMetrics(
        cyclomatic_complexity=visitor.complexity,
        max_nesting_depth=visitor._max_depth,
        branch_count=visitor.branch_count,
    )
