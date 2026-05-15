
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class JsonSerializable:
    def to_dict(self) -> dict:
        return _to_jsonable(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class CodeExample(JsonSerializable):
    code: str
    description: str | None = None
    source: Literal["docstring", "readme"] = "docstring"


@dataclass
class ParameterInfo(JsonSerializable):
    name: str
    annotation: str | None = None
    default: str | None = None
    kind: str = "positional"
    description: str | None = None


@dataclass
class ReturnInfo(JsonSerializable):
    annotation: str | None = None
    description: str | None = None


@dataclass
class RaiseInfo(JsonSerializable):
    exception: str
    description: str | None = None


@dataclass
class FunctionInfo(JsonSerializable):
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
    module: str | None = None
    names: list[str] = field(default_factory=list)
    lineno: int = 0


@dataclass
class ModuleInfo(JsonSerializable):
    name: str
    path: str
    docstring: str | None = None
    imports: list[ImportInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    examples: list[CodeExample] = field(default_factory=list)


@dataclass
class LibraryMetadata(JsonSerializable):
    name: str
    version: str | None = None
    license: str | None = None
    release_date: datetime | None = None
    source_type: Literal["local_path", "git", "pypi"] = "local_path"
    root_path: str = ""
    readme_path: str | None = None


@dataclass
class LibrarySnapshot(JsonSerializable):
    metadata: LibraryMetadata
    modules: list[ModuleInfo] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_dict(cls, data: dict) -> "LibrarySnapshot":
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
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_json_file(cls, path: str | Path) -> "LibrarySnapshot":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _to_jsonable(value):
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
    if not value:
        return None
    return datetime.fromisoformat(value)


def _module_from_dict(data: dict) -> ModuleInfo:
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
    return ParameterInfo(
        name=data["name"],
        annotation=data.get("annotation"),
        default=data.get("default"),
        kind=data.get("kind", "positional"),
        description=data.get("description"),
    )


def _example_from_dict(data: dict) -> CodeExample:
    return CodeExample(
        code=data["code"],
        description=data.get("description"),
        source=data.get("source", "docstring"),
    )
