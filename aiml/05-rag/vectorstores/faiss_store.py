"""
FAISS vector store — fast local similarity search.
Install: pip install faiss-cpu   or   pip install faiss-gpu
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ingestion.document_loader import Document
from ingestion.embedder import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    In-memory FAISS index with disk persistence.

    Example:
        store = FAISSVectorStore(embedder=embedder, index_type="Flat")
        store.add_documents(chunks)
        store.save("./faiss_index")

        # Later:
        store = FAISSVectorStore.load("./faiss_index", embedder=embedder)
        results = store.similarity_search("query", top_k=5)
    """

    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        index_type: str = "Flat",  # Flat | IVFFlat | HNSW
        nlist: int = 100,  # IVF clusters
    ) -> None:
        try:
            import faiss

            self._faiss = faiss
        except ImportError:
            raise ImportError("faiss required: pip install faiss-cpu")

        self._embedder = embedder
        self._index_type = index_type
        self._nlist = nlist
        self._index = None
        self._docs: List[Document] = []
        self._dimension = embedder.dimension

    def _make_index(self):
        if self._index_type == "Flat":
            return self._faiss.IndexFlatIP(
                self._dimension
            )  # Inner product (cosine if normalized)
        elif self._index_type == "IVFFlat":
            quantizer = self._faiss.IndexFlatIP(self._dimension)
            index = self._faiss.IndexIVFFlat(quantizer, self._dimension, self._nlist)
            return index
        elif self._index_type == "HNSW":
            return self._faiss.IndexHNSWFlat(self._dimension, 32)
        else:
            raise ValueError(f"Unknown FAISS index type: {self._index_type}")

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        texts = [d.page_content for d in docs]
        embeddings = self._embedder.embed(texts).astype(np.float32)
        self._faiss.normalize_L2(embeddings)

        if self._index is None:
            self._index = self._make_index()
            if hasattr(self._index, "train"):
                self._index.train(embeddings)

        self._index.add(embeddings)
        self._docs.extend(docs)
        logger.info("FAISS: %d documents added (total: %d)", len(docs), len(self._docs))

    def similarity_search(
        self, query: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        if self._index is None or len(self._docs) == 0:
            return []
        qvec = self._embedder.embed_query(query).astype(np.float32).reshape(1, -1)
        self._faiss.normalize_L2(qvec)
        scores, indices = self._index.search(qvec, min(top_k, len(self._docs)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            results.append((self._docs[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "docs.pkl", "wb") as f:
            pickle.dump(self._docs, f)
        logger.info("FAISS index saved to %s", path)

    @classmethod
    def load(
        cls, path: str, embedder: SentenceTransformerEmbedder, **kwargs
    ) -> "FAISSVectorStore":
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss required: pip install faiss-cpu")
        path = Path(path)
        instance = cls(embedder=embedder, **kwargs)
        instance._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "docs.pkl", "rb") as f:
            instance._docs = pickle.load(f)
        logger.info("FAISS index loaded from %s (%d docs)", path, len(instance._docs))
        return instance

    def count(self) -> int:
        return len(self._docs)
