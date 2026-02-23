"""
Cross-encoder reranker — upgrades coarse retrieval results with precise relevance scoring.
Install: pip install sentence-transformers
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from ingestion.document_loader import Document

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.

    Example:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, docs, top_n=3)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers required: pip install sentence-transformers"
            )

        self._model = CrossEncoder(model_name, device=device)
        self._batch_size = batch_size
        logger.info("CrossEncoderReranker loaded: %s on %s", model_name, device)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Score each document against the query and return sorted (doc, score) pairs.

        Args:
            query: The user question / query string.
            documents: Retrieved candidate documents.
            top_n: Limit output to top_n documents; returns all if None.

        Returns:
            List of (Document, score) sorted by descending relevance.
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        logger.debug(
            "CrossEncoderReranker: reranked %d → %d docs", len(documents), len(ranked)
        )
        return [(doc, float(score)) for doc, score in ranked]

    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Document],
        threshold: float = 0.1,
        top_n: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """Rerank and filter below a minimum relevance threshold."""
        ranked = self.rerank(query, documents, top_n=top_n)
        return [(doc, score) for doc, score in ranked if score >= threshold]
