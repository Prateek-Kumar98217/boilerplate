"""
Pydantic Settings v2 configuration for all inference clients.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    api_keys: List[str] = Field(default_factory=list, alias="GEMINI_API_KEYS")
    default_model: str = Field("gemini-2.0-flash", alias="GEMINI_DEFAULT_MODEL")
    vision_model: str = Field("gemini-2.0-flash", alias="GEMINI_VISION_MODEL")
    embedding_model: str = Field("text-embedding-004", alias="GEMINI_EMBEDDING_MODEL")
    rpm_limit: int = Field(15, alias="GEMINI_RPM_LIMIT")
    rpd_limit: int = Field(1500, alias="GEMINI_RPD_LIMIT")

    @field_validator("api_keys", mode="before")
    @classmethod
    def split_keys(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v or []


class GroqSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    api_keys: List[str] = Field(default_factory=list, alias="GROQ_API_KEYS")
    default_model: str = Field("llama-3.3-70b-versatile", alias="GROQ_DEFAULT_MODEL")
    whisper_model: str = Field("whisper-large-v3-turbo", alias="GROQ_WHISPER_MODEL")
    rpm_limit: int = Field(30, alias="GROQ_RPM_LIMIT")
    tpm_limit: int = Field(131072, alias="GROQ_TPM_LIMIT")

    @field_validator("api_keys", mode="before")
    @classmethod
    def split_keys(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v or []


class CerebrasSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    api_keys: List[str] = Field(default_factory=list, alias="CEREBRAS_API_KEYS")
    default_model: str = Field("llama3.1-70b", alias="CEREBRAS_DEFAULT_MODEL")
    rpm_limit: int = Field(30, alias="CEREBRAS_RPM_LIMIT")

    @field_validator("api_keys", mode="before")
    @classmethod
    def split_keys(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v or []


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    short_term_turns: int = Field(20, alias="MEMORY_SHORT_TERM_TURNS")
    max_short_tokens: int = Field(4096, alias="MEMORY_MAX_SHORT_TOKENS")
    summary_model: str = Field("", alias="MEMORY_SUMMARY_MODEL")


class PromptSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    default_system_prompt: str = Field(
        "You are a helpful, accurate, and concise AI assistant.",
        alias="DEFAULT_SYSTEM_PROMPT",
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )

    gemini: GeminiSettings = GeminiSettings()
    groq: GroqSettings = GroqSettings()
    cerebras: CerebrasSettings = CerebrasSettings()
    memory: MemorySettings = MemorySettings()
    prompt: PromptSettings = PromptSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
