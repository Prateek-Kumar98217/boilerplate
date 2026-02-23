"""
Deployment section config — Pydantic Settings.
Values are loaded from environment variables / .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    name: str = Field("ml-api", alias="APP_NAME")
    env: str = Field("development", alias="APP_ENV")
    host: str = Field("0.0.0.0", alias="APP_HOST")
    port: int = Field(8000, alias="APP_PORT")
    workers: int = Field(1, alias="APP_WORKERS")
    log_level: str = Field("info", alias="LOG_LEVEL")
    allowed_origins: List[str] = Field(default=["*"], alias="ALLOWED_ORIGINS")
    api_key: str = Field("", alias="API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


class ModelSettings(BaseSettings):
    model_dir: Path = Field(Path("./models/weights"), alias="MODEL_DIR")
    default_model: str = Field("my_model_v1", alias="DEFAULT_MODEL")
    max_batch_size: int = Field(32, alias="MAX_BATCH_SIZE")
    inference_timeout: int = Field(30, alias="INFERENCE_TIMEOUT")
    hf_token: str = Field("", alias="HF_TOKEN")
    hf_cache_dir: str = Field("~/.cache/huggingface", alias="HF_CACHE_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


class LLMProviderSettings(BaseSettings):
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    cerebras_api_key: str = Field("", alias="CEREBRAS_API_KEY")
    google_api_key: str = Field("", alias="GOOGLE_API_KEY")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


class TrackingSettings(BaseSettings):
    mlflow_tracking_uri: str = Field(
        "http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )
    wandb_api_key: str = Field("", alias="WANDB_API_KEY")
    wandb_project: str = Field("my-project", alias="WANDB_PROJECT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


class MonitoringSettings(BaseSettings):
    enable_metrics: bool = Field(True, alias="ENABLE_METRICS")
    metrics_port: int = Field(9090, alias="METRICS_PORT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


class Settings(BaseSettings):
    """Aggregate settings — single import point."""

    app: AppSettings = AppSettings()
    model: ModelSettings = ModelSettings()
    llm: LLMProviderSettings = LLMProviderSettings()
    tracking: TrackingSettings = TrackingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
