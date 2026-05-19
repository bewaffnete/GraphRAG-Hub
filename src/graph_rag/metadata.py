from __future__ import annotations

import json
import email
from pathlib import Path

from .models import LibraryMetadata


def detect_library_metadata(root: Path) -> LibraryMetadata:
    """
    Detect library metadata by inspecting project files like pyproject.toml, 
    setup.cfg, or package.json. If these are missing, it attempts to 
    read installed distribution info.

    Args:
        root (Path): The root directory of the library to inspect.

    Returns:
        LibraryMetadata: An object containing name, version, license, and paths.
    """
    pyproject_path = root / "pyproject.toml"
    setup_cfg_path = root / "setup.cfg"
    package_json_path = root / "package.json"
    readme_path = _detect_readme(root)

    name = root.name
    version = None
    license_name = None

    if pyproject_path.exists():
        try:
            import tomllib

            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            project = data.get("project", {})
            name = project.get("name", name)
            version = project.get("version")
            license_field = project.get("license")
            if isinstance(license_field, str):
                license_name = license_field
            elif isinstance(license_field, dict):
                license_name = license_field.get("text") or license_field.get("file")
        except Exception:
            pass
    elif package_json_path.exists():
        try:
            data = json.loads(package_json_path.read_text(encoding="utf-8"))
            name = data.get("name", name)
            version = data.get("version")
            license_name = data.get("license")
        except Exception:
            pass
    elif setup_cfg_path.exists():
        try:
            import configparser

            config = configparser.ConfigParser()
            config.read(setup_cfg_path, encoding="utf-8")
            if config.has_section("metadata"):
                name = config.get("metadata", "name", fallback=name)
                version = config.get("metadata", "version", fallback=None)
                license_name = config.get("metadata", "license", fallback=None)
        except Exception:
            pass

    if version is None or license_name is None:
        dist_info = _detect_dist_info(root)
        if dist_info:
            dist_name, dist_version, dist_license = _read_dist_info_metadata(dist_info)
            name = dist_name or name
            version = version or dist_version
            license_name = license_name or dist_license

    return LibraryMetadata(
        name=name,
        version=version,
        license=license_name,
        root_path=str(root),
        readme_path=str(readme_path) if readme_path else None,
    )


def _detect_readme(root: Path) -> Path | None:
    """
    Find the README file in the root directory.

    Args:
        root (Path): Directory to search.

    Returns:
        Path | None: Path to the README file if found, otherwise None.
    """
    candidates = sorted(root.glob("README*"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _detect_dist_info(root: Path) -> Path | None:
    """
    Try to find the .dist-info directory for a given library root.

    Args:
        root (Path): The library source root.

    Returns:
        Path | None: Path to the .dist-info directory if found.
    """
    parent = root.parent
    normalized = root.name.replace("-", "_").lower()
    for candidate in sorted(parent.glob("*.dist-info")):
        base_name = candidate.name.split("-", 1)[0].replace("-", "_").lower()
        if base_name == normalized:
            return candidate
        top_level = _read_top_level_modules(candidate)
        if normalized in top_level:
            return candidate
        record_modules = _read_record_top_level_modules(candidate)
        if normalized in record_modules:
            return candidate
    return None


def _read_dist_info_metadata(dist_info: Path) -> tuple[str | None, str | None, str | None]:
    """
    Read Name, Version, and License from a .dist-info/METADATA file.

    Args:
        dist_info (Path): Path to the .dist-info directory.

    Returns:
        tuple: (name, version, license) strings, any of which can be None.
    """
    metadata_path = dist_info / "METADATA"
    if not metadata_path.exists():
        return None, None, None
    try:
        msg = email.message_from_string(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None
    name = msg.get("Name")
    version = msg.get("Version")
    license_name = msg.get("License")
    return name, version, license_name


def _read_top_level_modules(dist_info: Path) -> set[str]:
    """
    Read top-level module names from .dist-info/top_level.txt.

    Args:
        dist_info (Path): Path to the .dist-info directory.

    Returns:
        set[str]: Set of module names.
    """
    top_level_path = dist_info / "top_level.txt"
    if not top_level_path.exists():
        return set()
    try:
        return {
            line.strip().replace("-", "_").lower()
            for line in top_level_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except Exception:
        return set()


def _read_record_top_level_modules(dist_info: Path) -> set[str]:
    """
    Extract top-level module names by parsing .dist-info/RECORD.

    Args:
        dist_info (Path): Path to the .dist-info directory.

    Returns:
        set[str]: Set of discovered top-level module names.
    """
    record_path = dist_info / "RECORD"
    if not record_path.exists():
        return set()
    modules: set[str] = set()
    try:
        for line in record_path.read_text(encoding="utf-8").splitlines():
            relative = line.split(",", 1)[0].strip()
            if not relative or relative.startswith(f"{dist_info.name}/"):
                continue
            top_level = relative.split("/", 1)[0].strip().replace("-", "_").lower()
            if top_level and not top_level.endswith(".dist-info"):
                modules.add(top_level.removesuffix(".py"))
    except Exception:
        return set()
    return modules
