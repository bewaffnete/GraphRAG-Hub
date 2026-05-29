"""Metadata extraction helpers for public API filtering."""

import ast
from configparser import ConfigParser
from email.parser import Parser
from pathlib import Path
import re
import tomllib


def extract_module_all(module: ast.Module) -> set[str]:
    exported: set[str] = set()
    for stmt in module.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in stmt.targets):
            continue
        value = stmt.value
        if isinstance(value, (ast.List, ast.Tuple)):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    exported.add(elt.value)
    return exported


def is_public_name(name: str) -> bool:
    return not name.startswith("_")


def module_is_public(module_name: str) -> bool:
    return all(part == "__init__" or not part.startswith("_") for part in module_name.split("."))


def detect_library_identity(root_path: Path) -> tuple[str | None, str | None]:
    root_path = root_path.resolve()
    name, version = _from_pyproject(root_path)
    if name or version:
        return name, version

    name, version = _from_setup_cfg(root_path)
    if name or version:
        return name, version

    name, version = _from_dist_info(root_path)
    if name or version:
        return name, version

    name, version = _from_init_version(root_path)
    return name, version


def _from_pyproject(root_path: Path) -> tuple[str | None, str | None]:
    pyproject = root_path / "pyproject.toml"
    if not pyproject.exists():
        return None, None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    project = data.get("project", {})
    name = project.get("name")
    version = project.get("version")
    return _normalize_name(name), _normalize_version(version)


def _from_setup_cfg(root_path: Path) -> tuple[str | None, str | None]:
    setup_cfg = root_path / "setup.cfg"
    if not setup_cfg.exists():
        return None, None
    parser = ConfigParser()
    parser.read(setup_cfg, encoding="utf-8")
    if not parser.has_section("metadata"):
        return None, None
    name = parser.get("metadata", "name", fallback=None)
    version = parser.get("metadata", "version", fallback=None)
    return _normalize_name(name), _normalize_version(version)


def _from_dist_info(root_path: Path) -> tuple[str | None, str | None]:
    parent = root_path.parent
    candidates = list(parent.glob("*.dist-info"))
    normalized_dir_name = _normalized_distribution_name(root_path.name)
    for dist_info in candidates:
        metadata_file = dist_info / "METADATA"
        if not metadata_file.exists():
            continue
        top_level_file = dist_info / "top_level.txt"
        top_levels = {line.strip() for line in top_level_file.read_text(encoding="utf-8").splitlines() if line.strip()} if top_level_file.exists() else set()
        if top_levels and root_path.name not in top_levels and normalized_dir_name not in {_normalized_distribution_name(item) for item in top_levels}:
            continue
        try:
            metadata = Parser().parsestr(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = metadata.get("Name")
        version = metadata.get("Version")
        if name or version:
            return _normalize_name(name), _normalize_version(version)
    return None, None


def _from_init_version(root_path: Path) -> tuple[str | None, str | None]:
    init_path = root_path / "__init__.py"
    if not init_path.exists():
        package_dirs = [path for path in root_path.iterdir() if path.is_dir() and (path / "__init__.py").exists()]
        init_path = (package_dirs[0] / "__init__.py") if package_dirs else init_path
    if not init_path.exists():
        return None, None
    try:
        content = init_path.read_text(encoding="utf-8")
    except Exception:
        return None, None
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
    package_name = init_path.parent.name
    return _normalize_name(package_name), _normalize_version(version_match.group(1) if version_match else None)


def _normalize_name(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_version(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().lower())
