"""RAG section config — Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    device: str = Field("auto", alias="EMBEDDING_DEVICE")
    batch_size: int = Field(64, alias="EMBEDDING_BATCH_SIZE")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class ChunkingSettings(BaseSettings):
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")
    strategy: str = Field("recursive", alias="CHUNKING_STRATEGY")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class ChromaSettings(BaseSettings):
    persist_dir: Path = Field(Path("./chroma_db"), alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field("rag_docs", alias="CHROMA_COLLECTION_NAME")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class FAISSSettings(BaseSettings):
    index_path: Path = Field(Path("./faiss_index"), alias="FAISS_INDEX_PATH")
    index_type: str = Field("Flat", alias="FAISS_INDEX_TYPE")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class PineconeSettings(BaseSettings):
    api_key: str = Field("", alias="PINECONE_API_KEY")
    index_name: str = Field("rag-index", alias="PINECONE_INDEX_NAME")
    cloud: str = Field("aws", alias="PINECONE_CLOUD")
    region: str = Field("us-east-1", alias="PINECONE_REGION")
    namespace: str = Field("default", alias="PINECONE_NAMESPACE")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class RetrievalSettings(BaseSettings):
    top_k: int = Field(5, alias="TOP_K")
    similarity_threshold: float = Field(0.0, alias="SIMILARITY_THRESHOLD")
    use_reranker: bool = Field(False, alias="USE_RERANKER")
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class LLMSettings(BaseSettings):
    provider: str = Field("groq", alias="LLM_PROVIDER")
    model: str = Field("llama-3.1-70b-versatile", alias="LLM_MODEL")
    temperature: float = Field(0.1, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(2048, alias="LLM_MAX_TOKENS")
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    google_api_key: str = Field("", alias="GOOGLE_API_KEY")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    cerebras_api_key: str = Field("", alias="CEREBRAS_API_KEY")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env", populate_by_name=True, extra="ignore"
    )


class Settings(BaseSettings):
    embedding: EmbeddingSettings = EmbeddingSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    chroma: ChromaSettings = ChromaSettings()
    faiss: FAISSSettings = FAISSSettings()
    pinecone: PineconeSettings = PineconeSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    llm: LLMSettings = LLMSettings()

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
