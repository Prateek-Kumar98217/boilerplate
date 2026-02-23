"""
Configuration for 06-agents using Pydantic Settings v2.
All values are read from environment variables / .env file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    provider: str = Field("groq", alias="LLM_PROVIDER")

    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-70b-versatile", alias="GROQ_MODEL")

    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-1.5-flash", alias="GEMINI_MODEL")

    cerebras_api_key: str = Field("", alias="CEREBRAS_API_KEY")
    cerebras_model: str = Field("llama3.1-70b", alias="CEREBRAS_MODEL")

    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(None, alias="OPENAI_BASE_URL")

    ollama_host: str = Field("http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field("llama3.2", alias="OLLAMA_MODEL")


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    max_iterations: int = Field(10, alias="AGENT_MAX_ITERATIONS")
    verbose: bool = Field(True, alias="AGENT_VERBOSE")
    stream: bool = Field(False, alias="AGENT_STREAM")


class ToolSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    serpapi_api_key: str = Field("", alias="SERPAPI_API_KEY")
    tavily_api_key: str = Field("", alias="TAVILY_API_KEY")
    code_execution_timeout: int = Field(10, alias="CODE_EXECUTION_TIMEOUT")
    tool_call_retries: int = Field(2, alias="TOOL_CALL_RETRIES")


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    memory_type: Literal["buffer", "summary", "vector"] = Field(
        "buffer", alias="MEMORY_TYPE"
    )
    max_tokens: int = Field(4096, alias="MEMORY_MAX_TOKENS")
    embedding_model: str = Field("all-MiniLM-L6-v2", alias="MEMORY_EMBEDDING_MODEL")
    vector_store_dir: str = Field("./agent_memory", alias="MEMORY_VECTOR_STORE_DIR")


class OrchestrationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    supervisor_model: str = Field("", alias="SUPERVISOR_MODEL")
    max_workers: int = Field(4, alias="MAX_WORKERS")
    handoff_timeout: int = Field(30, alias="HANDOFF_TIMEOUT")

    langsmith_api_key: str = Field("", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field("agents-boilerplate", alias="LANGSMITH_PROJECT")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    llm: LLMSettings = LLMSettings()
    agent: AgentSettings = AgentSettings()
    tool: ToolSettings = ToolSettings()
    memory: MemorySettings = MemorySettings()
    orchestration: OrchestrationSettings = OrchestrationSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
