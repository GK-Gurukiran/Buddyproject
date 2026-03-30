"""
Application configuration using Pydantic Settings.
Loads values from .env file with sensible defaults.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # --- Application ---
    app_name: str = "Scalable AI Chatbot Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    secret_key: str = "change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///./chatbot.db"

    # --- LLM Provider ---
    llm_provider: str = "ollama"  # "ollama" (free/local), "openai", or "openrouter"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "chatbot_memory"

    # --- Mem0 ---
    mem0_api_key: str = ""

    # --- Tavily ---
    tavily_api_key: str = ""

    # --- Firecrawl ---
    firecrawl_api_key: str = ""

    # --- LiveKit ---
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    livekit_agent_name: str = "buddy-voice-agent"
    deepgram_api_key: str = ""
    cartesia_api_key: str = ""

    # --- LangSmith ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "scalable-ai-chatbot"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
