
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class JsonSerializable:
    """Base class providing JSON serialization capabilities to dataclasses."""

    def to_dict(self) -> dict:
        """
        Convert the object to a JSON-serializable dictionary.

        Returns:
            dict: A dictionary representation of the object.
        """
        return _to_jsonable(self)

    def to_json(self, *, indent: int = 2) -> str:
        """
        Convert the object to a JSON string.

        Args:
            indent (int): The indentation level for the JSON output. Defaults to 2.

        Returns:
            str: A JSON string representation of the object.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class CodeExample(JsonSerializable):
    """
    Represents a code example extracted from documentation or source code.

    Attributes:
        code (str): The actual code snippet.
        description (str | None): An optional description of what the example demonstrates.
        source (Literal["docstring", "readme"]): The source of the example.
    """
    code: str
    description: str | None = None
    source: Literal["docstring", "readme"] = "docstring"


@dataclass
class ParameterInfo(JsonSerializable):
    """
    Metadata about a function or method parameter.

    Attributes:
        name (str): Parameter name.
        annotation (str | None): Type hint/annotation as a string.
        default (str | None): Default value as a string representation.
        kind (str): Kind of parameter (e.g., 'positional', 'keyword_only').
        description (str | None): Docstring description of the parameter.
    """
    name: str
    annotation: str | None = None
    default: str | None = None
    kind: str = "positional"
    description: str | None = None


@dataclass
class ReturnInfo(JsonSerializable):
    """
    Metadata about a function or method return value.

    Attributes:
        annotation (str | None): Return type hint/annotation as a string.
        description (str | None): Docstring description of the return value.
    """
    annotation: str | None = None
    description: str | None = None


@dataclass
class RaiseInfo(JsonSerializable):
    """
    Metadata about an exception a function or method might raise.

    Attributes:
        exception (str): The type of exception.
        description (str | None): Description of when or why it's raised.
    """
    exception: str
    description: str | None = None


@dataclass
class FunctionInfo(JsonSerializable):
    """
    Comprehensive metadata about a function or method.

    Attributes:
        name (str): Short name of the function.
        qualname (str): Fully qualified name.
        module (str): Module where it is defined.
        lineno (int): Starting line number in the source file.
        end_lineno (int | None): Ending line number.
        signature (str): Full string representation of the signature.
        docstring (str | None): Raw docstring content.
        decorators (list[str]): List of applied decorators.
        parameters (list[ParameterInfo]): Detailed parameter metadata.
        returns (ReturnInfo | None): Return value metadata.
        raises (list[RaiseInfo]): Documented exceptions.
        examples (list[CodeExample]): Extracted code examples.
        is_async (bool): Whether the function is defined with 'async def'.
        is_method (bool): Whether it is a class or instance method.
        logic_skeleton (str | None): A simplified version of the function's logic.
        internal_calls (list[str]): Names of internal functions called.
        constants_used (list[str]): Names of constants referenced.
        complexity_metrics (dict): Computed complexity scores (e.g., cyclomatic).
    """
    name: str
    qualname: str
    module: str
    lineno: int
    end_lineno: int | None = None
    signature: str = ""
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    parameters: list[ParameterInfo] = field(default_factory=list)
    returns: ReturnInfo | None = None
    raises: list[RaiseInfo] = field(default_factory=list)
    examples: list[CodeExample] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    logic_skeleton: str | None = None
    internal_calls: list[str] = field(default_factory=list)
    constants_used: list[str] = field(default_factory=list)
    complexity_metrics: dict = field(default_factory=dict)


@dataclass
class ClassInfo(JsonSerializable):
    """
    Comprehensive metadata about a Python class.

    Attributes:
        name (str): Name of the class.
        qualname (str): Fully qualified name.
        module (str): Module where it is defined.
        lineno (int): Starting line number in the source file.
        end_lineno (int | None): Ending line number.
        docstring (str | None): Raw docstring content.
        bases (list[str]): Names of base classes.
        decorators (list[str]): List of applied decorators.
        methods (list[FunctionInfo]): Metadata for methods defined in the class.
        examples (list[CodeExample]): Extracted code examples.
        internal_calls (list[str]): Internal calls made within the class scope.
        constants_used (list[str]): Constants referenced within the class scope.
        complexity_metrics (dict): Computed complexity scores.
    """
    name: str
    qualname: str
    module: str
    lineno: int
    end_lineno: int | None = None
    docstring: str | None = None
    bases: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    methods: list[FunctionInfo] = field(default_factory=list)
    examples: list[CodeExample] = field(default_factory=list)
    internal_calls: list[str] = field(default_factory=list)
    constants_used: list[str] = field(default_factory=list)
    complexity_metrics: dict = field(default_factory=dict)


@dataclass
class ImportInfo(JsonSerializable):
    """
    Represents an import statement in a module.

    Attributes:
        module (str | None): The module being imported from.
        names (list[str]): Specific names imported (e.g., ['func1', 'ClassA']).
        lineno (int): Line number where the import occurs.
    """
    module: str | None = None
    names: list[str] = field(default_factory=list)
    lineno: int = 0


@dataclass
class ModuleInfo(JsonSerializable):
    """
    Metadata for an entire Python module/file.

    Attributes:
        name (str): Dot-separated module name.
        path (str): Relative path to the source file.
        docstring (str | None): Module-level docstring.
        imports (list[ImportInfo]): List of imports found in the module.
        classes (list[ClassInfo]): Classes defined in the module.
        functions (list[FunctionInfo]): Functions defined at the module level.
        examples (list[CodeExample]): Extracted module-level examples.
    """
    name: str
    path: str
    docstring: str | None = None
    imports: list[ImportInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    examples: list[CodeExample] = field(default_factory=list)


@dataclass
class LibraryMetadata(JsonSerializable):
    """
    General metadata about a library or project.

    Attributes:
        name (str): Name of the library.
        version (str | None): Version string.
        license (str | None): License identifier.
        release_date (datetime | None): Date of release.
        source_type (Literal["local_path", "git", "pypi"]): Type of source.
        root_path (str): Root filesystem path.
        readme_path (str | None): Path to the README file.
    """
    name: str
    version: str | None = None
    license: str | None = None
    release_date: datetime | None = None
    source_type: Literal["local_path", "git", "pypi"] = "local_path"
    root_path: str = ""
    readme_path: str | None = None


@dataclass
class LibrarySnapshot(JsonSerializable):
    """
    A full snapshot of a library's structure and metadata.

    Attributes:
        metadata (LibraryMetadata): Project-level metadata.
        modules (list[ModuleInfo]): List of all modules within the library.
        extracted_at (datetime): Timestamp when the snapshot was generated.
    """
    metadata: LibraryMetadata
    modules: list[ModuleInfo] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_dict(cls, data: dict) -> "LibrarySnapshot":
        """
        Create a LibrarySnapshot instance from a dictionary.

        Args:
            data (dict): Dictionary containing snapshot data.

        Returns:
            LibrarySnapshot: The hydrated snapshot object.
        """
        metadata = LibraryMetadata(
            name=data["metadata"]["name"],
            version=data["metadata"].get("version"),
            license=data["metadata"].get("license"),
            release_date=_parse_datetime(data["metadata"].get("release_date")),
            source_type=data["metadata"].get("source_type", "local_path"),
            root_path=data["metadata"].get("root_path", ""),
            readme_path=data["metadata"].get("readme_path"),
        )
        modules = [_module_from_dict(module) for module in data.get("modules", [])]
        extracted_at = _parse_datetime(data.get("extracted_at")) or datetime.utcnow()
        return cls(metadata=metadata, modules=modules, extracted_at=extracted_at)

    @classmethod
    def from_json(cls, payload: str) -> "LibrarySnapshot":
        """
        Create a LibrarySnapshot instance from a JSON string.

        Args:
            payload (str): JSON string containing snapshot data.

        Returns:
            LibrarySnapshot: The hydrated snapshot object.
        """
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_json_file(cls, path: str | Path) -> "LibrarySnapshot":
        """
        Load a LibrarySnapshot from a JSON file.

        Args:
            path (str | Path): Path to the JSON file.

        Returns:
            LibrarySnapshot: The hydrated snapshot object.
        """
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _to_jsonable(value):
    """
    Recursively convert a value to a JSON-serializable format.

    Args:
        value: The value to convert.

    Returns:
        The JSON-serializable representation of the value.
    """
    if isinstance(value, JsonSerializable):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(inner) for inner in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _parse_datetime(value: str | None) -> datetime | None:
    """
    Parse an ISO format datetime string.

    Args:
        value (str | None): The datetime string to parse.

    Returns:
        datetime | None: The parsed datetime object, or None if input is empty.
    """
    if not value:
        return None
    return datetime.fromisoformat(value)


def _module_from_dict(data: dict) -> ModuleInfo:
    """Hydrate a ModuleInfo object from a dictionary."""
    return ModuleInfo(
        name=data["name"],
        path=data["path"],
        docstring=data.get("docstring"),
        imports=[
            ImportInfo(
                module=item.get("module"),
                names=item.get("names", []),
                lineno=item.get("lineno", 0),
            )
            for item in data.get("imports", [])
        ],
        classes=[_class_from_dict(item) for item in data.get("classes", [])],
        functions=[_function_from_dict(item) for item in data.get("functions", [])],
        examples=[_example_from_dict(item) for item in data.get("examples", [])],
    )


def _class_from_dict(data: dict) -> ClassInfo:
    """Hydrate a ClassInfo object from a dictionary."""
    return ClassInfo(
        name=data["name"],
        qualname=data["qualname"],
        module=data["module"],
        lineno=data["lineno"],
        end_lineno=data.get("end_lineno"),
        docstring=data.get("docstring"),
        bases=data.get("bases", []),
        decorators=data.get("decorators", []),
        methods=[_function_from_dict(item) for item in data.get("methods", [])],
        examples=[_example_from_dict(item) for item in data.get("examples", [])],
        internal_calls=data.get("internal_calls", []),
        constants_used=data.get("constants_used", []),
        complexity_metrics=data.get("complexity_metrics", {}),
    )


def _function_from_dict(data: dict) -> FunctionInfo:
    """Hydrate a FunctionInfo object from a dictionary."""
    returns = data.get("returns")
    return FunctionInfo(
        name=data["name"],
        qualname=data["qualname"],
        module=data["module"],
        lineno=data["lineno"],
        end_lineno=data.get("end_lineno"),
        signature=data.get("signature", ""),
        docstring=data.get("docstring"),
        decorators=data.get("decorators", []),
        parameters=[_parameter_from_dict(item) for item in data.get("parameters", [])],
        returns=ReturnInfo(**returns) if returns else None,
        raises=[RaiseInfo(**item) for item in data.get("raises", [])],
        examples=[_example_from_dict(item) for item in data.get("examples", [])],
        is_async=data.get("is_async", False),
        is_method=data.get("is_method", False),
        logic_skeleton=data.get("logic_skeleton"),
        internal_calls=data.get("internal_calls", []),
        constants_used=data.get("constants_used", []),
        complexity_metrics=data.get("complexity_metrics", {}),
    )


def _parameter_from_dict(data: dict) -> ParameterInfo:
    """Hydrate a ParameterInfo object from a dictionary."""
    return ParameterInfo(
        name=data["name"],
        annotation=data.get("annotation"),
        default=data.get("default"),
        kind=data.get("kind", "positional"),
        description=data.get("description"),
    )


def _example_from_dict(data: dict) -> CodeExample:
    """Hydrate a CodeExample object from a dictionary."""
    return CodeExample(
        code=data["code"],
        description=data.get("description"),
        source=data.get("source", "docstring"),
    )
