"""Interactive configuration utilities for the Graph RAG CLI."""

import os
from pathlib import Path
import importlib.metadata
import importlib.util

try:
    import questionary
    from questionary import Style
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from dotenv import set_key, load_dotenv
except ImportError:
    raise ImportError("Interactive CLI requires 'questionary', 'rich', and 'python-dotenv'. Install with: pip install -e '.[cli]'")

# Custom style for questionary to fix visibility issues (using black for maximum contrast)
custom_style = Style([
    ('qmark', 'fg:#000000 bold'),       # question mark
    ('question', 'fg:#000000 bold'),    # question text
    ('answer', 'fg:#000000 bold'),      # submitted answer text
    ('pointer', 'fg:#000000 bold'),     # pointer used in select and checkbox
    ('highlighted', 'fg:#000000 bold'), # pointed-at choice in select and checkbox
    ('selected', 'fg:#000000'),         # (checkbox) checked item
    ('separator', 'fg:#000000'),        # separator in lists
    ('instruction', 'fg:#000000'),      # user instructions
    ('text', 'fg:#000000'),             # plain text
    ('disabled', 'fg:#858585 italic')   # disabled choices
])

ENV_FILE = Path(".env")
console = Console()

KNOWN_GEMINI_MODELS = ["gemini-embedding-001", "gemini-embedding-2"]
KNOWN_OLLAMA_MODELS = ["embeddinggemma", "qwen3-embedding", "all-minilm"]

def _ensure_env_file() -> None:
    """Create a .env file if it doesn't exist."""
    if not ENV_FILE.exists():
        ENV_FILE.touch()

def _save_env(key: str, value: str) -> None:
    """Save a key-value pair to the .env file and update current environment."""
    _ensure_env_file()
    set_key(str(ENV_FILE), key, value)
    os.environ[key] = value

def setup_neo4j_interactive() -> None:
    """Interactive wizard for Neo4j connection settings."""
    console.print("[cyan]Neo4j Configuration[/cyan]")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    uri = questionary.text("Neo4j URI:", default=uri, style=custom_style).ask()
    if not uri: return
    _save_env("NEO4J_URI", uri)

    username = questionary.text("Neo4j Username:", default=username, style=custom_style).ask()
    if not username: return
    _save_env("NEO4J_USERNAME", username)

    password = questionary.password("Neo4j Password:", style=custom_style).ask()
    if password:
        _save_env("NEO4J_PASSWORD", password)
    else:
        password = os.getenv("NEO4J_PASSWORD")

    database = questionary.text("Neo4j Database:", default=database, style=custom_style).ask()
    if not database: return
    _save_env("NEO4J_DATABASE", database)

    if questionary.confirm("Test connection?", style=custom_style).ask():
        console.print("[cyan]Testing Neo4j connection...[/cyan]")
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=(username, password), notifications_min_severity="OFF")
            driver.verify_connectivity()
            console.print("[bold green]Connection successful![/bold green]")
            driver.close()
        except Exception as e:
            console.print(f"[bold red]Connection failed: {e}[/bold red]")


def setup_embedding_interactive() -> None:
    """Interactive wizard for Embedding provider settings."""
    console.print("[cyan]Embedding Provider Configuration[/cyan]")
    provider = os.getenv("EMBEDDING_PROVIDER", "hash")
    provider = questionary.select(
        "Select Embedding Provider:",
        choices=["hash", "openai", "gemini", "ollama"],
        default=provider,
        style=custom_style
    ).ask()
    if not provider: return
    
    _save_env("EMBEDDING_PROVIDER", provider)

    if provider == "openai":
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        model = questionary.text("OpenAI Model:", default=model, style=custom_style).ask()
        if model: _save_env("EMBEDDING_MODEL", model)
        
        api_key = questionary.password("OpenAI API Key:", style=custom_style).ask()
        if api_key: _save_env("OPENAI_API_KEY", api_key)
        
    elif provider == "gemini":
        model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        model = questionary.autocomplete(
            "Gemini Model:",
            choices=KNOWN_GEMINI_MODELS,
            default=model,
            style=custom_style
        ).ask()
        if model: _save_env("EMBEDDING_MODEL", model)
        
        api_key = questionary.password("Gemini API Key:", style=custom_style).ask()
        if api_key: _save_env("GEMINI_API_KEY", api_key)
        
    elif provider == "ollama":
        model = os.getenv("EMBEDDING_MODEL", "embeddinggemma")
        model = questionary.autocomplete(
            "Ollama Model:",
            choices=KNOWN_OLLAMA_MODELS,
            default=model,
            style=custom_style
        ).ask()
        if model: _save_env("EMBEDDING_MODEL", model)
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        base_url = questionary.text("Ollama Base URL:", default=base_url, style=custom_style).ask()
        if base_url: _save_env("OLLAMA_BASE_URL", base_url)
        
    elif provider == "hash":
        console.print("[italic]Hash provider selected (local fallback, no API keys required).[/italic]")
        _save_env("EMBEDDING_MODEL", "hash-embedding-v1")


def get_installed_packages() -> list[str]:
    """Get a list of names of all installed pip packages."""
    return sorted(list({dist.metadata["Name"] for dist in importlib.metadata.distributions() if dist.metadata["Name"]}))


def select_installed_package_interactive() -> str | None:
    """Prompt user to select an installed package and resolve its physical path."""
    packages = get_installed_packages()
    choices = ["[Enter manual path]"] + packages
    
    selection = questionary.autocomplete(
        "Search or select a library to ingest:",
        choices=choices,
        style=custom_style
    ).ask()
    
    if not selection:
        return None
        
    if selection == "[Enter manual path]":
        path = questionary.path("Path to local repository or package root:", style=custom_style).ask()
        return path
        
    # Resolve physical path of the selected package
    console.print(f"[cyan]Resolving path for package '{selection}'...[/cyan]")
    try:
        spec = importlib.util.find_spec(selection.replace("-", "_"))
        if spec and spec.submodule_search_locations:
            path = spec.submodule_search_locations[0]
            console.print(f"[green]Resolved path: {path}[/green]")
            return path
        elif spec and spec.origin:
            path = str(Path(spec.origin).parent)
            console.print(f"[green]Resolved path: {path}[/green]")
            return path
    except Exception:
        pass
        
    console.print(f"[yellow]Could not automatically resolve physical path for '{selection}'.[/yellow]")
    path = questionary.path("Please enter path manually:", style=custom_style).ask()
    return path


def print_config_status() -> None:
    """Print the current Graph RAG configuration in a table."""
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    
    neo4j_uri = os.getenv("NEO4J_URI", "Not set")
    neo4j_user = os.getenv("NEO4J_USERNAME", "Not set")
    provider = os.getenv("EMBEDDING_PROVIDER", "Not set")
    model = os.getenv("EMBEDDING_MODEL", "Not set")
    
    table.add_row("Neo4j URI:", neo4j_uri)
    table.add_row("Neo4j User:", neo4j_user)
    table.add_row("Provider:", provider)
    table.add_row("Model:", model)
    
    console.print(Panel(table, title="[bold]Graph RAG Configuration[/bold]", expand=False))
