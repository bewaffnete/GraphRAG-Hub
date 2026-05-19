import os
from typing import Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(temperature: float = 0, json_mode: bool = False) -> BaseChatModel:
    """
    Factory function to get a LangChain LLM based on environment variables.

    Supported providers (set via LLM_PROVIDER):
    - 'ollama': Requires OLLAMA_BASE_URL (defaults to localhost).
    - 'openai': Requires OPENAI_API_KEY.
    - 'anthropic': Requires ANTHROPIC_API_KEY.
    - 'gemini': Requires GEMINI_API_KEY.

    Args:
        temperature (float): Sampling temperature. Defaults to 0.
        json_mode (bool): Whether to request JSON output from the model.

    Returns:
        BaseChatModel: A configured LangChain chat model instance.

    Raises:
        ValueError: If the specified LLM_PROVIDER is unsupported.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("LLM_MODEL")

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or "gemma4:31b-cloud",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            format="json" if json_mode else None
        )
    
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            temperature=temperature,
            model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {}
        )
    
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Anthropic handle JSON via system prompt usually, but we return the model
        return ChatAnthropic(
            model=model or "claude-3-5-sonnet-20240620",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature
        )
    
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-pro",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
