"""Interactive configuration utilities and onboarding wizard for Graph RAG."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import urlopen

from .env import load_app_env

try:
    import questionary
    from dotenv import set_key
    from questionary import Style
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    raise ImportError(
        "Interactive CLI requires 'questionary', 'rich', and 'python-dotenv'. "
        "Install with: pip install -e '.[cli]'"
    )

custom_style = Style(
    [
        ("qmark", "fg:#000000 bold"),
        ("question", "fg:#000000 bold"),
        ("answer", "fg:#000000 bold"),
        ("pointer", "fg:#000000 bold"),
        ("highlighted", "fg:#000000 bold"),
        ("selected", "fg:#000000"),
        ("separator", "fg:#000000"),
        ("instruction", "fg:#000000"),
        ("text", "fg:#000000"),
        ("disabled", "fg:#858585 italic"),
    ]
)

ENV_FILE = Path(".env")
console = Console()

KNOWN_GEMINI_MODELS = ["gemini-embedding-001", "gemini-embedding-2"]
KNOWN_OLLAMA_MODELS = ["embeddinggemma", "qwen3-embedding", "all-minilm"]
KNOWN_OPENAI_MODELS = ["text-embedding-3-large", "text-embedding-3-small"]
VENV_DIR_NAMES = (".venv", "venv", "env")
OLLAMA_LOCAL_URL = "http://localhost:11434"
OLLAMA_HOST_DOCKER_URL = "http://host.docker.internal:11434"
OLLAMA_DOCKER_URL = "http://ollama:11434"


@dataclass(slots=True)
class VenvCandidate:
    """Represents a discovered virtual environment."""

    root: Path
    site_packages: Path | None
    python_version: str | None


@dataclass(slots=True)
class SetupDiagnostics:
    """Represents preflight diagnostics for the onboarding wizard."""

    docker_installed: bool
    docker_compose_available: bool
    docker_path: str | None
    compose_command: list[str] | None
    repo_root: Path
    env_file: Path
    venv_candidates: list[VenvCandidate]
    local_ollama_available: bool


def _ensure_env_file() -> None:
    """Create a .env file if it doesn't exist."""
    if not ENV_FILE.exists():
        ENV_FILE.touch()


def _save_env(key: str, value: str) -> None:
    """Save a key-value pair to the .env file and update current environment."""
    _ensure_env_file()
    set_key(str(ENV_FILE), key, value)
    os.environ[key] = value


def _strip_or_none(value: str | None) -> str | None:
    """Normalize user input by stripping whitespace and collapsing empties."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _default_neo4j_password() -> str:
    """Generate a secure default password for local Neo4j."""
    return secrets.token_urlsafe(18)


def _mask_secret(value: str | None) -> str:
    """Mask a secret for display."""
    if not value:
        return "not set"
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def _read_python_version(venv_root: Path) -> str | None:
    """Best-effort extraction of the Python version used by a venv."""
    cfg_path = venv_root / "pyvenv.cfg"
    if cfg_path.exists():
        for line in cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("version"):
                return line.split("=", 1)[-1].strip()

    lib_dir = venv_root / "lib"
    if lib_dir.exists():
        for child in lib_dir.iterdir():
            if child.is_dir() and child.name.startswith("python"):
                return child.name.removeprefix("python")
    return None


def _resolve_site_packages(venv_root: Path) -> Path | None:
    """Find the site-packages path for a venv on Unix-like systems."""
    lib_dir = venv_root / "lib"
    if not lib_dir.exists():
        return None

    for child in sorted(lib_dir.iterdir()):
        site_packages = child / "site-packages"
        if child.is_dir() and child.name.startswith("python") and site_packages.exists():
            return site_packages
    return None


def _within_depth(base: Path, candidate: Path, max_depth: int) -> bool:
    """Return True when candidate is within max_depth relative to base."""
    try:
        relative = candidate.relative_to(base)
    except ValueError:
        return False
    return len(relative.parts) <= max_depth


def discover_virtualenvs(repo_root: Path, max_depth: int = 2) -> list[VenvCandidate]:
    """Search nearby directories for virtual environments and site-packages paths."""
    roots_to_scan = [repo_root]
    roots_to_scan.extend(repo_root.parents[:2])

    found: dict[Path, VenvCandidate] = {}
    for root in roots_to_scan:
        if not root.exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            current = Path(dirpath)
            if not _within_depth(root, current, max_depth):
                dirnames[:] = []
                continue

            for dirname in list(dirnames):
                if dirname not in VENV_DIR_NAMES:
                    continue
                venv_root = current / dirname
                resolved = venv_root.resolve()
                found[resolved] = VenvCandidate(
                    root=resolved,
                    site_packages=_resolve_site_packages(resolved),
                    python_version=_read_python_version(resolved),
                )
                dirnames.remove(dirname)

    return sorted(found.values(), key=lambda item: (str(item.root)))


def is_ollama_available(url: str = OLLAMA_LOCAL_URL) -> bool:
    """Check whether an Ollama server responds at the given base URL."""
    endpoint = f"{url.rstrip('/')}/api/tags"
    try:
        with urlopen(endpoint, timeout=2) as response:
            return 200 <= response.status < 500
    except URLError:
        return False


def run_host_diagnostics(repo_root: Path | None = None) -> SetupDiagnostics:
    """Inspect the host for docker tooling and candidate virtual environments."""
    resolved_root = (repo_root or Path.cwd()).resolve()
    docker_path = shutil.which("docker")

    docker_installed = docker_path is not None
    compose_command: list[str] | None = None
    docker_compose_available = False

    if docker_installed:
        compose_process = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if compose_process.returncode == 0:
            docker_compose_available = True
            compose_command = ["docker", "compose"]

    return SetupDiagnostics(
        docker_installed=docker_installed,
        docker_compose_available=docker_compose_available,
        docker_path=docker_path,
        compose_command=compose_command,
        repo_root=resolved_root,
        env_file=resolved_root / ".env",
        venv_candidates=discover_virtualenvs(resolved_root),
        local_ollama_available=is_ollama_available(),
    )


def _prompt_existing_or_generated_secret(prompt: str, env_key: str, default_factory: Callable[[], str]) -> str:
    """Prompt for a secret, prefilled by an existing env value when available."""
    existing = _strip_or_none(os.getenv(env_key))
    default_value = existing or default_factory()
    value = questionary.password(
        prompt,
        default=default_value,
        style=custom_style,
    ).ask()
    return _strip_or_none(value) or default_value


def _prompt_validated_path(prompt: str, default: str | None = None, required: bool = True) -> str | None:
    """Prompt for a filesystem path and validate existence immediately."""
    while True:
        value = questionary.path(prompt, default=default, style=custom_style).ask()
        value = _strip_or_none(value)
        if not value:
            if required:
                console.print("[bold red]A path is required.[/bold red]")
                continue
            return None

        if Path(value).exists():
            return value

        console.print(f"[yellow]Path does not exist: {value}[/yellow]")
        if not questionary.confirm("Try another path?", default=True, style=custom_style).ask():
            return value


def _select_venv_path(diagnostics: SetupDiagnostics) -> str | None:
    """Choose the host site-packages path mounted into the app container."""
    existing = _strip_or_none(os.getenv("VENV_LIBS_PATH"))
    choices: list[str] = []
    mapping: dict[str, str | None] = {}

    if existing:
        label = f"Use existing VENV_LIBS_PATH ({existing})"
        choices.append(label)
        mapping[label] = existing

    for candidate in diagnostics.venv_candidates:
        if not candidate.site_packages:
            continue
        python_version = candidate.python_version or "unknown"
        label = f"{candidate.site_packages} (venv: {candidate.root.name}, python {python_version})"
        choices.append(label)
        mapping[label] = str(candidate.site_packages)

    manual_label = "Enter path manually"
    skip_label = "Skip venv mount for now"
    choices.extend([manual_label, skip_label])
    mapping[manual_label] = None
    mapping[skip_label] = None

    selection = questionary.select(
        "Choose the host site-packages path to mount into the container:",
        choices=choices,
        style=custom_style,
    ).ask()
    if not selection or selection == skip_label:
        return None
    if selection == manual_label:
        return _prompt_validated_path("Path to site-packages directory:", default=existing, required=False)
    return mapping[selection]


def _default_provider() -> str:
    """Pick the best default embedding provider for a local-first setup."""
    provider = _strip_or_none(os.getenv("EMBEDDING_PROVIDER"))
    if provider:
        return provider
    return "ollama"


def _prompt_ollama_base_url(diagnostics: SetupDiagnostics) -> str:
    """Prompt for the Ollama base URL using the three supported runtime modes."""
    existing = _strip_or_none(os.getenv("OLLAMA_BASE_URL"))
    default_url = existing or (OLLAMA_HOST_DOCKER_URL if diagnostics.local_ollama_available else OLLAMA_DOCKER_URL)

    status = "detected on localhost:11434" if diagnostics.local_ollama_available else "not detected on localhost:11434"
    console.print(f"[cyan]Ollama status:[/cyan] {status}")

    choices = [
        f"Local GraphRAG launch on this machine -> {OLLAMA_LOCAL_URL}",
        f"Docker GraphRAG + local host Ollama -> {OLLAMA_HOST_DOCKER_URL}",
        f"Docker GraphRAG + Docker Ollama service -> {OLLAMA_DOCKER_URL}",
        "Enter URL manually",
    ]
    default_choice = next((choice for choice in choices if default_url in choice), None)
    if default_choice is None:
        default_choice = choices[1] if diagnostics.local_ollama_available else choices[2]

    selection = questionary.select(
        "How should GraphRAG connect to Ollama?",
        choices=choices,
        default=default_choice,
        style=custom_style,
    ).ask()
    selection = _strip_or_none(selection)
    if not selection:
        return default_url
    if selection == "Enter URL manually":
        return (
            _strip_or_none(
                questionary.text(
                    "Ollama base URL:",
                    default=default_url,
                    style=custom_style,
                ).ask()
            )
            or default_url
        )
    return selection.split(" -> ", 1)[1].strip()


def _prompt_provider_config(diagnostics: SetupDiagnostics) -> dict[str, str]:
    """Prompt for the embedding provider and its required settings."""
    provider = questionary.select(
        "Select the embedding provider:",
        choices=["ollama", "openai", "gemini", "hash"],
        default=_default_provider(),
        style=custom_style,
    ).ask()
    provider = _strip_or_none(provider) or "hash"

    config = {"EMBEDDING_PROVIDER": provider}
    if provider == "openai":
        model = questionary.autocomplete(
            "OpenAI embedding model:",
            choices=KNOWN_OPENAI_MODELS,
            default=_strip_or_none(os.getenv("EMBEDDING_MODEL")) or "text-embedding-3-large",
            style=custom_style,
        ).ask()
        api_key = questionary.password(
            "OpenAI API key:",
            default=_strip_or_none(os.getenv("OPENAI_API_KEY")),
            style=custom_style,
        ).ask()
        config["EMBEDDING_MODEL"] = _strip_or_none(model) or "text-embedding-3-large"
        if _strip_or_none(api_key):
            config["OPENAI_API_KEY"] = _strip_or_none(api_key) or ""
        config["OPENAI_BASE_URL"] = _strip_or_none(os.getenv("OPENAI_BASE_URL")) or "https://api.openai.com/v1"
    elif provider == "gemini":
        model = questionary.autocomplete(
            "Gemini embedding model:",
            choices=KNOWN_GEMINI_MODELS,
            default=_strip_or_none(os.getenv("EMBEDDING_MODEL")) or "gemini-embedding-001",
            style=custom_style,
        ).ask()
        api_key = questionary.password(
            "Gemini API key:",
            default=_strip_or_none(os.getenv("GEMINI_API_KEY")),
            style=custom_style,
        ).ask()
        config["EMBEDDING_MODEL"] = _strip_or_none(model) or "gemini-embedding-001"
        if _strip_or_none(api_key):
            config["GEMINI_API_KEY"] = _strip_or_none(api_key) or ""
        config["GEMINI_BASE_URL"] = (
            _strip_or_none(os.getenv("GEMINI_BASE_URL"))
            or "https://generativelanguage.googleapis.com/v1beta"
        )
    elif provider == "ollama":
        model = questionary.autocomplete(
            "Ollama embedding model:",
            choices=KNOWN_OLLAMA_MODELS,
            default=_strip_or_none(os.getenv("EMBEDDING_MODEL")) or "embeddinggemma",
            style=custom_style,
        ).ask()
        llm_model = questionary.text(
            "Default chat model for LangGraph agent:",
            default=_strip_or_none(os.getenv("LLM_MODEL")) or "gemma2:9b",
            style=custom_style,
        ).ask()
        base_url = _prompt_ollama_base_url(diagnostics)
        config["EMBEDDING_MODEL"] = _strip_or_none(model) or "embeddinggemma"
        config["OLLAMA_BASE_URL"] = _strip_or_none(base_url) or OLLAMA_HOST_DOCKER_URL
        config["LLM_PROVIDER"] = "ollama"
        config["LLM_MODEL"] = _strip_or_none(llm_model) or "gemma2:9b"
    else:
        config["EMBEDDING_MODEL"] = "hash-embedding-v1"

    config["EMBEDDING_DIMENSIONS"] = _strip_or_none(os.getenv("EMBEDDING_DIMENSIONS")) or "256"
    return config


def _build_env_config(diagnostics: SetupDiagnostics) -> dict[str, str]:
    """Collect wizard answers and build the resulting .env mapping."""
    console.print("[cyan]Phase 1/5[/cyan] Host diagnostics complete.")

    venv_path = _select_venv_path(diagnostics)
    console.print("[cyan]Phase 2/5[/cyan] Configure local services.")

    neo4j_username = _strip_or_none(os.getenv("NEO4J_USERNAME")) or "neo4j"
    neo4j_password = _prompt_existing_or_generated_secret(
        "Neo4j password:",
        "NEO4J_PASSWORD",
        _default_neo4j_password,
    )
    neo4j_database = _strip_or_none(os.getenv("NEO4J_DATABASE")) or "neo4j"
    neo4j_uri = _strip_or_none(os.getenv("NEO4J_URI")) or "bolt://localhost:7687"

    console.print("[cyan]Phase 3/5[/cyan] Choose your embedding provider.")
    provider_config = _prompt_provider_config(diagnostics)

    env_config = {
        "NEO4J_URI": neo4j_uri,
        "NEO4J_USERNAME": neo4j_username,
        "NEO4J_PASSWORD": neo4j_password,
        "NEO4J_DATABASE": neo4j_database,
        **provider_config,
    }
    if venv_path:
        env_config["VENV_LIBS_PATH"] = venv_path

    return env_config


def _render_diagnostics_panel(diagnostics: SetupDiagnostics) -> None:
    """Display a concise preflight summary."""
    table = Table(show_header=False, box=None)
    table.add_column("Check", style="cyan")
    table.add_column("Result", style="green")
    table.add_row("Repository", str(diagnostics.repo_root))
    table.add_row("Docker", "available" if diagnostics.docker_installed else "missing")
    table.add_row("Docker Compose", "available" if diagnostics.docker_compose_available else "missing")
    table.add_row("Local Ollama", "available" if diagnostics.local_ollama_available else "missing")
    table.add_row("Detected venvs", str(len(diagnostics.venv_candidates)))
    console.print(Panel(table, title="[bold]Host Diagnostics[/bold]", expand=False))


def _render_env_review(env_config: dict[str, str]) -> None:
    """Display the final configuration before writing .env."""
    table = Table(show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    for key in sorted(env_config):
        value = env_config[key]
        display_value = _mask_secret(value) if "KEY" in key or "PASSWORD" in key else value
        table.add_row(key, display_value)

    console.print(Panel(table, title="[bold]Configuration Review[/bold]", expand=False))


def write_env_config(env_config: dict[str, str]) -> Path:
    """Persist the generated environment configuration to .env."""
    _ensure_env_file()
    for key, value in env_config.items():
        if value:
            _save_env(key, value)
    return ENV_FILE.resolve()


def wait_for_http_ok(url: str, timeout_seconds: float = 60.0, interval_seconds: float = 2.0) -> bool:
    """Poll a URL until it responds successfully or a timeout is reached."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return True
        except URLError:
            pass
        time.sleep(interval_seconds)
    return False


def bootstrap_docker_services(diagnostics: SetupDiagnostics, env_config: dict[str, str]) -> bool:
    """Run docker compose up -d and wait for Neo4j to become reachable."""
    if not diagnostics.docker_installed:
        console.print("[bold red]Docker is not installed. Install Docker Desktop and rerun setup.[/bold red]")
        return False
    if not diagnostics.docker_compose_available or not diagnostics.compose_command:
        console.print("[bold red]`docker compose` is unavailable. Please install or enable Docker Compose.[/bold red]")
        return False
    if env_config.get("OLLAMA_BASE_URL") == OLLAMA_LOCAL_URL:
        console.print(
            "[bold yellow]Current Ollama URL is set for local CLI execution (`http://localhost:11434`). "
            "If you plan to run GraphRAG inside Docker, rerun setup and choose either "
            "`http://host.docker.internal:11434` or `http://ollama:11434`.[/bold yellow]"
        )
        return False

    compose_command = [*diagnostics.compose_command]
    if env_config.get("OLLAMA_BASE_URL") == OLLAMA_DOCKER_URL:
        compose_command.extend(["--profile", "with-ollama"])

    with console.status("[bold cyan]Spawning Docker containers...[/bold cyan]", spinner="dots"):
        process = subprocess.run(
            [*compose_command, "up", "-d"],
            cwd=diagnostics.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        console.print("[bold red]Docker startup failed.[/bold red]")
        if process.stderr.strip():
            console.print(process.stderr.strip())
        return False

    with console.status("[bold cyan]Waiting for Neo4j health check...[/bold cyan]", spinner="dots"):
        ready = wait_for_http_ok("http://localhost:7474", timeout_seconds=90.0)

    if not ready:
        console.print("[bold yellow]Containers started, but Neo4j did not become healthy in time.[/bold yellow]")
        return False

    console.print(
        Panel.fit(
            "Configuration complete!\n"
            "Neo4j is reachable at http://localhost:7474\n"
            "Next step: `docker compose exec graph-rag-app graph-rag ingest`",
            title="[bold green]Success[/bold green]",
        )
    )
    return True


def run_setup_wizard(repo_root: Path | None = None) -> None:
    """Run the frictionless onboarding flow described in TODO.md."""
    load_app_env()
    diagnostics = run_host_diagnostics(repo_root)

    console.print(Panel.fit("[bold cyan]GraphRAG-Hub Setup Wizard[/bold cyan]"))
    _render_diagnostics_panel(diagnostics)

    if not diagnostics.docker_installed:
        console.print(
            "[yellow]Docker is not installed, so the wizard can still generate `.env`, "
            "but it cannot boot the stack yet.[/yellow]"
        )

    env_config = _build_env_config(diagnostics)

    console.print("[cyan]Phase 4/5[/cyan] Review generated configuration.")
    _render_env_review(env_config)
    if not questionary.confirm("Write this configuration to .env?", default=True, style=custom_style).ask():
        console.print("[yellow]Setup cancelled before writing .env.[/yellow]")
        return

    env_path = write_env_config(env_config)
    console.print(f"[bold green]Saved configuration to {env_path}[/bold green]")

    console.print("[cyan]Phase 5/5[/cyan] Docker bootstrap.")
    if questionary.confirm("Would you like to start the Docker containers now?", default=True, style=custom_style).ask():
        bootstrap_docker_services(diagnostics, env_config)
    else:
        console.print("Run `docker compose up -d` when you're ready.")


def setup_neo4j_interactive() -> None:
    """Interactive wizard for Neo4j connection settings."""
    console.print("[cyan]Neo4j Configuration[/cyan]")
    uri = _strip_or_none(os.getenv("NEO4J_URI")) or "bolt://localhost:7687"
    username = _strip_or_none(os.getenv("NEO4J_USERNAME")) or "neo4j"
    database = _strip_or_none(os.getenv("NEO4J_DATABASE")) or "neo4j"

    uri = _strip_or_none(questionary.text("Neo4j URI:", default=uri, style=custom_style).ask())
    if not uri:
        return
    _save_env("NEO4J_URI", uri)

    username = _strip_or_none(questionary.text("Neo4j Username:", default=username, style=custom_style).ask())
    if not username:
        return
    _save_env("NEO4J_USERNAME", username)

    password = _strip_or_none(questionary.password("Neo4j Password:", style=custom_style).ask())
    if password:
        _save_env("NEO4J_PASSWORD", password)
    else:
        password = _strip_or_none(os.getenv("NEO4J_PASSWORD"))

    database = _strip_or_none(questionary.text("Neo4j Database:", default=database, style=custom_style).ask())
    if not database:
        return
    _save_env("NEO4J_DATABASE", database)

    if questionary.confirm("Test connection?", style=custom_style).ask():
        console.print("[cyan]Testing Neo4j connection...[/cyan]")
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(uri, auth=(username, password), notifications_min_severity="OFF")
            driver.verify_connectivity()
            console.print("[bold green]Connection successful![/bold green]")
            driver.close()
        except Exception as exc:
            console.print(f"[bold red]Connection failed: {exc}[/bold red]")


def setup_embedding_interactive() -> None:
    """Interactive wizard for Embedding provider settings."""
    console.print("[cyan]Embedding Provider Configuration[/cyan]")
    config = _prompt_provider_config()
    for key, value in config.items():
        _save_env(key, value)

    if config["EMBEDDING_PROVIDER"] == "hash":
        console.print("[italic]Hash provider selected (local fallback, no API keys required).[/italic]")


def get_installed_packages() -> list[str]:
    """Get names of installed packages from local metadata and mounted venv folders."""
    packages = set()
    for dist in importlib.metadata.distributions():
        if dist.metadata and dist.metadata["Name"]:
            packages.add(dist.metadata["Name"])

    user_venv_path = "/user_venv"
    if os.path.exists(user_venv_path):
        try:
            for dist in importlib.metadata.distributions(paths=[user_venv_path]):
                if dist.metadata and dist.metadata["Name"]:
                    packages.add(dist.metadata["Name"])
        except Exception:
            pass

        try:
            ignored_prefixes = (".", "__")
            ignored_suffixes = (".dist-info", ".egg-info", ".egg", "-info")
            ignored_names = {"bin", "include", "share", "lib", "lib64"}

            for item in os.listdir(user_venv_path):
                full_path = os.path.join(user_venv_path, item)
                if os.path.isdir(full_path):
                    if item.startswith(ignored_prefixes) or item.lower() in ignored_names:
                        continue
                    if any(item.endswith(suffix) for suffix in ignored_suffixes):
                        continue
                    packages.add(item)
                    packages.add(item.replace("_", "-"))
        except Exception:
            pass

    return sorted(packages)


def select_installed_package_interactive() -> str | None:
    """Prompt user to select an installed package and resolve its physical path."""
    packages = get_installed_packages()
    choices = ["[Enter manual path]"] + packages

    selection = questionary.autocomplete(
        "Search or select a library to ingest:",
        choices=choices,
        style=custom_style,
    ).ask()
    selection = _strip_or_none(selection)
    if not selection:
        return None

    if selection.startswith(("/", "./", "../")):
        console.print(f"[green]Detected direct path input: {selection}[/green]")
        return selection

    if selection == "[Enter manual path]":
        return _prompt_validated_path("Path to local repository or package root:", required=False)

    console.print(f"[cyan]Resolving path for package '{selection}'...[/cyan]")
    user_venv_path = "/user_venv"
    if os.path.exists(user_venv_path):
        candidate_names = [selection, selection.replace("-", "_"), selection.replace("_", "-")]
        for name in candidate_names:
            candidate_path = os.path.join(user_venv_path, name)
            if os.path.isdir(candidate_path):
                console.print(f"[green]Resolved path in user venv (folder): {candidate_path}[/green]")
                return candidate_path

        try:
            dists = [d for d in importlib.metadata.distributions(paths=[user_venv_path]) if d.metadata["Name"] == selection]
            if dists and dists[0].files:
                for file_path in dists[0].files:
                    parts = Path(file_path).parts
                    if len(parts) > 1 and not parts[0].endswith((".dist-info", ".egg-info")):
                        candidate_path = os.path.join(user_venv_path, parts[0])
                        if os.path.isdir(candidate_path):
                            console.print(f"[green]Resolved path in user venv (metadata): {candidate_path}[/green]")
                            return candidate_path
        except Exception:
            pass

    try:
        spec = importlib.util.find_spec(selection.replace("-", "_"))
        if spec and spec.submodule_search_locations:
            path = spec.submodule_search_locations[0]
            console.print(f"[green]Resolved path: {path}[/green]")
            return path
        if spec and spec.origin:
            path = str(Path(spec.origin).parent)
            console.print(f"[green]Resolved path: {path}[/green]")
            return path
    except Exception:
        pass

    console.print(f"[yellow]Could not automatically resolve physical path for '{selection}'.[/yellow]")
    return _prompt_validated_path("Please enter path manually:", required=False)


def print_config_status() -> None:
    """Print the current Graph RAG configuration in a table."""
    load_app_env()
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Neo4j URI:", os.getenv("NEO4J_URI", "Not set"))
    table.add_row("Neo4j User:", os.getenv("NEO4J_USERNAME", "Not set"))
    table.add_row("Provider:", os.getenv("EMBEDDING_PROVIDER", "Not set"))
    table.add_row("Model:", os.getenv("EMBEDDING_MODEL", "Not set"))
    table.add_row("Venv libs:", os.getenv("VENV_LIBS_PATH", "Not set"))

    console.print(Panel(table, title="[bold]Graph RAG Configuration[/bold]", expand=False))
