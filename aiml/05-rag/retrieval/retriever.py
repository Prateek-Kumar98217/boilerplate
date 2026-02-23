"""
Unified retriever — wraps Chroma / FAISS / Pinecone stores with optional BM25 hybrid search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ingestion.document_loader import Document

logger = logging.getLogger(__name__)

VectorStore = Any  # ChromaVectorStore | FAISSVectorStore | PineconeVectorStore


@dataclass
class RetrievalResult:
    document: Document
    score: float
    rank: int


class DenseRetriever:
    """Dense semantic retriever backed by any of the three vector stores."""

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        self._store = vector_store
        self._top_k = top_k
        self._threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        k = top_k or self._top_k
        raw: List[Tuple[Document, float]] = self._store.similarity_search(
            query, top_k=k, **({"filter": filter} if filter else {})
        )
        results = [
            RetrievalResult(document=doc, score=score, rank=rank)
            for rank, (doc, score) in enumerate(raw)
            if score >= self._threshold
        ]
        logger.debug("DenseRetriever: %d results for '%s'", len(results), query[:60])
        return results


class BM25Retriever:
    """Sparse BM25 retriever over an in-memory document corpus."""

    def __init__(self, top_k: int = 5) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self._bm25_cls = BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 required: pip install rank-bm25")

        self._top_k = top_k
        self._docs: List[Document] = []
        self._bm25 = None

    def index(self, docs: List[Document]) -> None:
        self._docs = docs
        tokenised = [d.page_content.lower().split() for d in docs]
        self._bm25 = self._bm25_cls(tokenised)
        logger.info("BM25Retriever: indexed %d documents", len(docs))

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        if self._bm25 is None:
            return []
        k = top_k or self._top_k
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :k
        ]
        return [
            RetrievalResult(document=self._docs[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        ]


class HybridRetriever:
    """
    Combines dense and sparse retrieval with Reciprocal Rank Fusion (RRF).

    RRF score: 1 / (rank + k) summed across retriever lists.
    """

    RRF_K = 60

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        top_k: int = 5,
        dense_weight: float = 0.7,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._top_k = top_k
        self._dense_weight = dense_weight

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        k = top_k or self._top_k
        dense_results = self._dense.retrieve(query, top_k=k * 2)
        sparse_results = self._sparse.retrieve(query, top_k=k * 2)

        # Build doc_id → score map using page_content as key
        rrf_scores: Dict[str, float] = {}
        rrf_docs: Dict[str, Document] = {}

        for weight, results in [
            (self._dense_weight, dense_results),
            (1.0 - self._dense_weight, sparse_results),
        ]:
            for res in results:
                key = res.document.page_content[:200]
                rrf_docs[key] = res.document
                rrf_scores[key] = rrf_scores.get(key, 0.0) + weight * (
                    1.0 / (res.rank + self.RRF_K)
                )

        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
        return [
            RetrievalResult(document=rrf_docs[key], score=rrf_scores[key], rank=rank)
            for rank, key in enumerate(sorted_keys)
        ]
