"""
Pinecone vector store — serverless cloud-scale similarity search.
Install: pip install pinecone-client
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ingestion.document_loader import Document
from ingestion.embedder import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)


class PineconeVectorStore:
    """
    Cloud-hosted Pinecone vector store with batched upsert.

    Example:
        store = PineconeVectorStore(
            api_key="...", index_name="rag-index",
            embedder=embedder, cloud="aws", region="us-east-1"
        )
        store.create_index_if_not_exists()
        store.add_documents(chunks)
        results = store.similarity_search("query", top_k=5)
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedder: SentenceTransformerEmbedder,
        cloud: str = "aws",
        region: str = "us-east-1",
        namespace: str = "default",
    ) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec

            self._pc_cls = Pinecone
            self._spec_cls = ServerlessSpec
        except ImportError:
            raise ImportError("pinecone required: pip install pinecone-client")

        self._api_key = api_key
        self._index_name = index_name
        self._embedder = embedder
        self._cloud = cloud
        self._region = region
        self._namespace = namespace
        self._pc = self._pc_cls(api_key=api_key)
        self._index = None

    def create_index_if_not_exists(self, metric: str = "cosine") -> None:
        """Create the Pinecone index if it does not exist."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name not in existing:
            self._pc.create_index(
                name=self._index_name,
                dimension=self._embedder.dimension,
                metric=metric,
                spec=self._spec_cls(cloud=self._cloud, region=self._region),
            )
            logger.info("Pinecone index '%s' created (%s)", self._index_name, metric)
            # Wait for index to be ready
            while not self._pc.describe_index(self._index_name).status["ready"]:
                time.sleep(1)
        else:
            logger.info("Pinecone index '%s' already exists", self._index_name)
        self._index = self._pc.Index(self._index_name)

    def add_documents(
        self,
        docs: List[Document],
        batch_size: int = 100,
        id_prefix: str = "doc",
    ) -> None:
        if self._index is None:
            self.create_index_if_not_exists()

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [d.page_content for d in batch]
            embeddings = self._embedder.embed(texts)

            vectors = []
            for j, (emb, doc) in enumerate(zip(embeddings, batch)):
                vectors.append(
                    {
                        "id": f"{id_prefix}_{i + j}",
                        "values": emb.tolist(),
                        "metadata": {**doc.metadata, "text": doc.page_content[:1000]},
                    }
                )
            self._index.upsert(vectors=vectors, namespace=self._namespace)

        logger.info(
            "Pinecone: upserted %d vectors to '%s'", len(docs), self._index_name
        )

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
        threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        if self._index is None:
            self._index = self._pc.Index(self._index_name)

        qvec = self._embedder.embed_query(query).tolist()
        response = self._index.query(
            vector=qvec,
            top_k=top_k,
            namespace=self._namespace,
            filter=filter,
            include_metadata=True,
        )
        results = []
        for match in response.matches:
            score = match.score
            if score < threshold:
                continue
            meta = dict(match.metadata)
            text = meta.pop("text", "")
            results.append((Document(page_content=text, metadata=meta), score))
        return results

    def delete_all(self) -> None:
        if self._index:
            self._index.delete(delete_all=True, namespace=self._namespace)
