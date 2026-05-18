import os
from pathlib import Path
import importlib.metadata
import importlib.util

try:
    import questionary
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from dotenv import set_key, load_dotenv
except ImportError:
    raise ImportError("Interactive CLI requires 'questionary', 'rich', and 'python-dotenv'. Install with: pip install -e '.[cli]'")

ENV_FILE = Path(".env")
console = Console()

KNOWN_GEMINI_MODELS = ["gemini-embedding-001", "gemini-embedding-2"]
KNOWN_OLLAMA_MODELS = ["embeddinggemma", "qwen3-embedding", "all-minilm"]

def _ensure_env_file() -> None:
    if not ENV_FILE.exists():
        ENV_FILE.touch()

def _save_env(key: str, value: str) -> None:
    _ensure_env_file()
    set_key(str(ENV_FILE), key, value)
    os.environ[key] = value

def setup_neo4j_interactive() -> None:
    console.print("[cyan]Neo4j Configuration[/cyan]")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    uri = questionary.text("Neo4j URI:", default=uri).ask()
    if not uri: return
    _save_env("NEO4J_URI", uri)

    username = questionary.text("Neo4j Username:", default=username).ask()
    if not username: return
    _save_env("NEO4J_USERNAME", username)

    password = questionary.password("Neo4j Password:").ask()
    if password:
        _save_env("NEO4J_PASSWORD", password)
    else:
        password = os.getenv("NEO4J_PASSWORD")

    database = questionary.text("Neo4j Database:", default=database).ask()
    if not database: return
    _save_env("NEO4J_DATABASE", database)

    if questionary.confirm("Test connection?").ask():
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
    console.print("[cyan]Embedding Provider Configuration[/cyan]")
    provider = os.getenv("EMBEDDING_PROVIDER", "hash")
    provider = questionary.select(
        "Select Embedding Provider:",
        choices=["hash", "openai", "gemini", "ollama"],
        default=provider
    ).ask()
    if not provider: return
    
    _save_env("EMBEDDING_PROVIDER", provider)

    if provider == "openai":
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        model = questionary.text("OpenAI Model:", default=model).ask()
        if model: _save_env("EMBEDDING_MODEL", model)
        
        api_key = questionary.password("OpenAI API Key:").ask()
        if api_key: _save_env("OPENAI_API_KEY", api_key)
        
    elif provider == "gemini":
        model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        model = questionary.autocomplete(
            "Gemini Model:",
            choices=KNOWN_GEMINI_MODELS,
            default=model
        ).ask()
        if model: _save_env("EMBEDDING_MODEL", model)
        
        api_key = questionary.password("Gemini API Key:").ask()
        if api_key: _save_env("GEMINI_API_KEY", api_key)
        
    elif provider == "ollama":
        model = os.getenv("EMBEDDING_MODEL", "embeddinggemma")
        model = questionary.autocomplete(
            "Ollama Model:",
            choices=KNOWN_OLLAMA_MODELS,
            default=model
        ).ask()
        if model: _save_env("EMBEDDING_MODEL", model)
        
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        base_url = questionary.text("Ollama Base URL:", default=base_url).ask()
        if base_url: _save_env("OLLAMA_BASE_URL", base_url)
        
    elif provider == "hash":
        console.print("[italic]Hash provider selected (local fallback, no API keys required).[/italic]")
        _save_env("EMBEDDING_MODEL", "hash-embedding-v1")


def get_installed_packages() -> list[str]:
    return sorted(list({dist.metadata["Name"] for dist in importlib.metadata.distributions() if dist.metadata["Name"]}))


def select_installed_package_interactive() -> str | None:
    packages = get_installed_packages()
    choices = ["[Enter manual path]"] + packages
    
    selection = questionary.autocomplete(
        "Search or select a library to ingest:",
        choices=choices
    ).ask()
    
    if not selection:
        return None
        
    if selection == "[Enter manual path]":
        path = questionary.path("Path to local repository or package root:").ask()
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
    path = questionary.path("Please enter path manually:").ask()
    return path


def print_config_status() -> None:
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

