"""
Online RAG pipeline — Pinecone vector store + Groq / Gemini / Cerebras LLMs.
Designed for production cloud deployments.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ingestion.document_loader import DirectoryLoader, Document
from ingestion.chunker import get_chunker
from ingestion.embedder import SentenceTransformerEmbedder
from vectorstores.pinecone_store import PineconeVectorStore
from retrieval.retriever import DenseRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.chain import create_llm_client, RAGChain, RAGResponse, LLMClient

logger = logging.getLogger(__name__)


@dataclass
class OnlineRAGConfig:
    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index: str = "rag-prod"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "default"
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    # Chunking
    chunk_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.0
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 3
    # LLM
    llm_provider: str = "groq"  # groq | gemini | cerebras | openai
    llm_api_key: str = ""
    llm_model: str = ""


class OnlineRAGPipeline:
    """
    Production RAG pipeline backed by Pinecone + cloud LLMs.

    Steps:
        1. ingest_directory() — load → chunk → embed → upsert to Pinecone
        2. query() — dense retrieve → (rerank) → generate
    """

    def __init__(self, config: Optional[OnlineRAGConfig] = None) -> None:
        self.cfg = config or OnlineRAGConfig(
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
        )
        self._embedder = SentenceTransformerEmbedder(
            model_name=self.cfg.embedding_model,
            device=self.cfg.embedding_device,
        )
        self._store = PineconeVectorStore(
            api_key=self.cfg.pinecone_api_key,
            index_name=self.cfg.pinecone_index,
            embedder=self._embedder,
            cloud=self.cfg.pinecone_cloud,
            region=self.cfg.pinecone_region,
            namespace=self.cfg.pinecone_namespace,
        )
        self._store.create_index_if_not_exists()
        self._chain: Optional[RAGChain] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_directory(self, directory: str, glob: str = "**/*") -> int:
        loader = DirectoryLoader(directory=directory, glob=glob)
        docs = loader.load()
        return self.ingest_documents(docs)

    def ingest_documents(self, docs: List[Document]) -> int:
        chunker = get_chunker(
            strategy=self.cfg.chunk_strategy,
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )
        chunks = chunker.split_documents(docs)
        self._store.add_documents(chunks)
        self._chain = None  # invalidate cached chain
        return len(chunks)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None,
    ) -> RAGResponse:
        chain = self._get_or_build_chain()
        return chain.query(question, top_k=top_k)

    def query_with_history(
        self,
        question: str,
        history: List[Dict[str, str]],
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        chain = self._get_or_build_chain()
        return chain.query_with_history(question, history=history, top_k=top_k)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_build_chain(self) -> RAGChain:
        if self._chain is not None:
            return self._chain

        retriever = DenseRetriever(
            vector_store=self._store,
            top_k=self.cfg.top_k,
            score_threshold=self.cfg.score_threshold,
        )
        llm = create_llm_client(
            provider=self.cfg.llm_provider,
            api_key=self.cfg.llm_api_key,
            model=self.cfg.llm_model,
        )
        self._chain = RAGChain(retriever=retriever, llm=llm, top_k=self.cfg.top_k)
        return self._chain
