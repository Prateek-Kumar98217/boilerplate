"""
Embedding models — wraps sentence-transformers and HuggingFace embeddings.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """
    Embed texts using sentence-transformers.
    Install: pip install sentence-transformers

    Example:
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        vecs = embedder.embed(["Hello world", "Another sentence"])
        # vecs.shape = (2, 384)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers required: pip install sentence-transformers"
            )

        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize = normalize
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedder: %s (dim=%d) on %s", model_name, self.dimension, device)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed one or multiple texts.

        Returns:
            np.ndarray of shape (N, D).
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 1000,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns (D,) array."""
        return self.embed([query])[0]

    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.embed(texts)


class LangChainEmbedder:
    """
    Thin wrapper to use any LangChain embeddings (OpenAI, Cohere, etc.)
    as a drop-in replacement.

    Example:
        from langchain_openai import OpenAIEmbeddings
        embedder = LangChainEmbedder(OpenAIEmbeddings())
    """

    def __init__(self, lc_embeddings) -> None:
        self._lc = lc_embeddings
        # Try to probe dimension
        sample = self._lc.embed_query("test")
        self.dimension = len(sample)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._lc.embed_documents(texts)
        return np.array(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return np.array(self._lc.embed_query(query), dtype=np.float32)
