"""
ChromaDB vector store wrapper — local persistent vector database.
Install: pip install chromadb
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ingestion.document_loader import Document
from ingestion.embedder import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Persistent ChromaDB vector store.

    Example:
        store = ChromaVectorStore(persist_dir="./chroma_db", embedder=embedder)
        store.add_documents(chunks)
        results = store.similarity_search("What is RAG?", top_k=5)
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_docs",
        embedder: Optional[SentenceTransformerEmbedder] = None,
    ) -> None:
        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb required: pip install chromadb")

        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB: collection '%s' at '%s' (%d docs)",
            collection_name,
            persist_dir,
            self._collection.count(),
        )

    def add_documents(self, docs: List[Document], batch_size: int = 100) -> None:
        """Embed and upsert documents."""
        if not docs:
            return
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [d.page_content for d in batch]
            metadatas = [d.metadata for d in batch]
            ids = [f"doc_{i + j}" for j in range(len(batch))]

            if self._embedder:
                embeddings = self._embedder.embed(texts).tolist()
                self._collection.upsert(
                    ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
                )
            else:
                self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

        logger.info("ChromaDB: added %d documents", len(docs))

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.

        Returns:
            List of (Document, score) tuples sorted by descending similarity.
        """
        query_kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": top_k}
        if self._embedder:
            query_emb = self._embedder.embed_query(query).tolist()
            query_kwargs = {"query_embeddings": [query_emb], "n_results": top_k}
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(
            **query_kwargs, include=["documents", "metadatas", "distances"]
        )
        docs_and_scores = []
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = 1.0 - dist  # Convert cosine distance → similarity
            if score >= threshold:
                docs_and_scores.append(
                    (Document(page_content=text, metadata=meta), score)
                )
        return docs_and_scores

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection.name)

    def count(self) -> int:
        return self._collection.count()
