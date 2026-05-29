from pathlib import Path

from graphrag_mcp.interfaces.cli.main import (
    _build_request,
    _default_library_scan_root,
    _discover_library_candidates,
    _filter_library_candidates,
    _looks_like_python_library,
    _resolve_library_scan_root,
)
from graphrag_mcp.infrastructure.parsing.metadata_extractor import detect_library_identity


def test_discover_library_candidates_filters_python_projects(tmp_path: Path) -> None:
    library_dir = tmp_path / "sample_lib"
    library_dir.mkdir()
    (library_dir / "pyproject.toml").write_text("[project]\nname='sample-lib'\n", encoding="utf-8")

    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "todo.txt").write_text("hi", encoding="utf-8")

    candidates = _discover_library_candidates(tmp_path)

    assert candidates == [library_dir]


def test_looks_like_python_library_supports_package_layout(tmp_path: Path) -> None:
    library_dir = tmp_path / "my_lib"
    package_dir = library_dir / "my_lib"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    assert _looks_like_python_library(library_dir) is True


def test_build_request_keeps_cli_values() -> None:
    request = _build_request(
        path="/tmp/lib",
        library_name="demo",
        version="1.2.3",
        ingest_mode="parse_load",
        embedding_mode="disabled",
    )

    assert request.path == "/tmp/lib"
    assert request.library_name == "demo"
    assert request.version == "1.2.3"


def test_resolve_library_scan_root_prefers_site_packages_in_env(tmp_path: Path) -> None:
    env_root = tmp_path / ".venv"
    site_packages = env_root / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)

    assert _resolve_library_scan_root(env_root) == site_packages


def test_filter_library_candidates_prioritizes_prefix_matches() -> None:
    candidates = [
        Path("/tmp/site-packages/annotated_types"),
        Path("/tmp/site-packages/numba"),
        Path("/tmp/site-packages/numpy"),
        Path("/tmp/site-packages/typing_extensions"),
    ]

    matches = _filter_library_candidates(candidates, "nu")

    assert matches == [
        Path("/tmp/site-packages/numba"),
        Path("/tmp/site-packages/numpy"),
    ]


def test_default_library_scan_root_uses_virtual_env(monkeypatch, tmp_path: Path) -> None:
    env_root = tmp_path / ".venv"
    site_packages = env_root / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setenv("VIRTUAL_ENV", str(env_root))

    assert _default_library_scan_root() == site_packages


def test_detect_library_identity_from_pyproject(tmp_path: Path) -> None:
    library_dir = tmp_path / "demo_lib"
    library_dir.mkdir()
    (library_dir / "pyproject.toml").write_text(
        "[project]\nname='demo-lib'\nversion='1.2.3'\n",
        encoding="utf-8",
    )

    name, version = detect_library_identity(library_dir)

    assert name == "demo-lib"
    assert version == "1.2.3"


def test_detect_library_identity_from_init_version(tmp_path: Path) -> None:
    library_dir = tmp_path / "demo_lib"
    package_dir = library_dir / "demo_lib"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("__version__ = '0.9.1'\n", encoding="utf-8")

    name, version = detect_library_identity(library_dir)

    assert name == "demo_lib"
    assert version == "0.9.1"
