import os
from typing import Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(temperature: float = 0, json_mode: bool = False) -> BaseChatModel:
    """
    Factory function to get a LangChain LLM based on environment variables.

    Only Ollama is supported for LLM.
    - 'ollama': Requires OLLAMA_BASE_URL (defaults to localhost).

    Args:
        temperature (float): Sampling temperature. Defaults to 0.
        json_mode (bool): Whether to request JSON output from the model.

    Returns:
        BaseChatModel: A configured LangChain chat model instance.

    Raises:
        ValueError: If a provider other than 'ollama' is specified.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("LLM_MODEL")

    if provider != "ollama":
        raise ValueError(f"Only 'ollama' is supported as an LLM provider. Requested: {provider}")

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model or "gemma4:31b-cloud",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
        format="json" if json_mode else None
    )
