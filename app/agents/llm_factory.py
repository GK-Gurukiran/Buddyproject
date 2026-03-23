"""
LLM Factory: Returns the correct LLM based on configuration.
Supports OpenAI (paid), OpenRouter (paid, multi-model), and Ollama (free/local).
"""

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from app.config import get_settings

settings = get_settings()


def get_llm(temperature: float = 0, streaming: bool = True):
    """
    Returns the configured LLM instance.
    - openrouter: Uses OpenRouter API (supports many models via single key)
    - openai: Uses direct OpenAI API
    - ollama: Free, runs locally
    """
    if settings.llm_provider == "openrouter" and settings.openrouter_api_key:
        return ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=temperature,
            streaming=streaming,
        )
    elif settings.llm_provider == "openai" and settings.openai_api_key:
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            streaming=streaming,
        )
    else:
        # Ollama - free, runs locally
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )
