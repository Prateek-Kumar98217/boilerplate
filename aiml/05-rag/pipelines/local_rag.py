"""
Local RAG pipeline — fully offline using ChromaDB / FAISS + sentence-transformers + Ollama.
No internet connection required after initial model downloads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from ingestion.document_loader import DirectoryLoader, Document
from ingestion.chunker import get_chunker
from ingestion.embedder import SentenceTransformerEmbedder
from vectorstores.chromadb_store import ChromaVectorStore
from vectorstores.faiss_store import FAISSVectorStore
from retrieval.retriever import DenseRetriever, BM25Retriever, HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.chain import OllamaClient, RAGChain, RAGResponse

logger = logging.getLogger(__name__)


class VectorBackend(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"


@dataclass
class LocalRAGConfig:
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    # Chunking
    chunk_strategy: str = "recursive"  # recursive | sentence | token
    chunk_size: int = 512
    chunk_overlap: int = 64
    # Vector store
    backend: VectorBackend = VectorBackend.CHROMA
    chroma_dir: str = "./chroma_local"
    chroma_collection: str = "local_rag"
    faiss_index_path: str = "./faiss_local.index"
    # Retrieval
    top_k: int = 5
    use_hybrid: bool = False
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 3
    score_threshold: float = 0.0
    # LLM
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"


class LocalRAGPipeline:
    """
    Complete local RAG pipeline.

    Steps:
        1. ingest_directory() — load → chunk → embed → store
        2. query() — retrieve (+ rerank) → generate
    """

    def __init__(self, config: Optional[LocalRAGConfig] = None) -> None:
        self.cfg = config or LocalRAGConfig()
        self._embedder = SentenceTransformerEmbedder(
            model_name=self.cfg.embedding_model,
            device=self.cfg.embedding_device,
        )
        self._store: ChromaVectorStore | FAISSVectorStore = self._build_store()
        self._retriever: DenseRetriever | HybridRetriever | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._chain: RAGChain | None = None

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def _build_store(self):
        if self.cfg.backend == VectorBackend.CHROMA:
            return ChromaVectorStore(
                collection_name=self.cfg.chroma_collection,
                embedder=self._embedder,
                persist_directory=self.cfg.chroma_dir,
            )
        return FAISSVectorStore(embedder=self._embedder)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_directory(self, directory: str, glob: str = "**/*") -> int:
        """Load all documents in a directory, chunk and index them."""
        loader = DirectoryLoader(directory=directory, glob=glob)
        docs = loader.load()
        logger.info("Loaded %d raw documents", len(docs))

        chunker = get_chunker(
            strategy=self.cfg.chunk_strategy,
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )
        chunks = chunker.split_documents(docs)
        logger.info("Created %d chunks", len(chunks))

        self._store.add_documents(chunks)
        self._reset_chain()
        return len(chunks)

    def ingest_documents(self, docs: List[Document]) -> int:
        """Ingest pre-loaded Document objects."""
        chunker = get_chunker(
            strategy=self.cfg.chunk_strategy,
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )
        chunks = chunker.split_documents(docs)
        self._store.add_documents(chunks)
        self._reset_chain()
        return len(chunks)

    def save_index(self, path: Optional[str] = None) -> None:
        """Persist FAISS index to disk (no-op for ChromaDB)."""
        if isinstance(self._store, FAISSVectorStore):
            self._store.save(path or self.cfg.faiss_index_path)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        chain = self._get_or_build_chain()
        return chain.query(question, top_k=top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_build_chain(self) -> RAGChain:
        if self._chain is not None:
            return self._chain

        dense = DenseRetriever(
            vector_store=self._store,
            top_k=self.cfg.top_k * (2 if self.cfg.use_hybrid else 1),
            score_threshold=self.cfg.score_threshold,
        )

        if self.cfg.use_hybrid:
            sparse = BM25Retriever(top_k=self.cfg.top_k * 2)
            retriever: DenseRetriever | HybridRetriever = HybridRetriever(
                dense=dense, sparse=sparse, top_k=self.cfg.top_k
            )
        else:
            retriever = dense

        if self.cfg.use_reranker:
            self._reranker = CrossEncoderReranker(
                model_name=self.cfg.reranker_model,
                device=self.cfg.embedding_device,
            )

        llm = OllamaClient(
            model=self.cfg.ollama_model,
            host=self.cfg.ollama_host,
        )
        self._chain = RAGChain(retriever=retriever, llm=llm, top_k=self.cfg.top_k)
        return self._chain

    def _reset_chain(self) -> None:
        self._chain = None
