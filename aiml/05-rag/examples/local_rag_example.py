"""
Local RAG example — fully offline pipeline using ChromaDB + Ollama.
Run:  python examples/local_rag_example.py
"""

from __future__ import annotations

import logging
import tempfile
import textwrap
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Sample documents (in a real scenario, point to your docs directory)
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = {
    "rag_intro.txt": textwrap.dedent(
        """\
        Retrieval-Augmented Generation (RAG) combines information retrieval with
        large language model generation. A document corpus is first indexed as
        dense embeddings in a vector store. At query time the most relevant
        chunks are retrieved and injected into the LLM prompt as context,
        grounding the generated answer in factual source material.
    """
    ),
    "faiss_notes.txt": textwrap.dedent(
        """\
        FAISS (Facebook AI Similarity Search) is a library for efficient dense
        vector search. It supports inner-product and L2 distance indices.
        The IndexFlatIP index computes exact inner products; when vectors are
        L2-normalised this is equivalent to cosine similarity.
    """
    ),
    "chromadb_notes.txt": textwrap.dedent(
        """\
        ChromaDB is an open-source embeddings database designed for AI applications.
        It stores document chunks alongside their embeddings and metadata in a
        persistent local directory, making it easy to run fully offline RAG
        pipelines without cloud infrastructure.
    """
    ),
}


def main() -> None:
    # --- write sample docs to a temp directory ---
    with tempfile.TemporaryDirectory() as tmpdir:
        doc_dir = Path(tmpdir)
        for name, content in SAMPLE_TEXTS.items():
            (doc_dir / name).write_text(content)

        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        from pipelines.local_rag import LocalRAGConfig, LocalRAGPipeline, VectorBackend

        # --- configure pipeline ---
        cfg = LocalRAGConfig(
            backend=VectorBackend.CHROMA,
            chroma_dir=str(doc_dir / "chroma"),
            embedding_model="all-MiniLM-L6-v2",
            ollama_model="llama3.2",
            chunk_size=256,
            chunk_overlap=32,
            top_k=3,
            use_hybrid=False,
            use_reranker=False,
        )
        pipeline = LocalRAGPipeline(config=cfg)

        # --- ingest ---
        n = pipeline.ingest_directory(str(doc_dir), glob="*.txt")
        print(f"\nIndexed {n} chunks from {len(SAMPLE_TEXTS)} documents.\n")

        # --- query ---
        questions = [
            "What is Retrieval-Augmented Generation?",
            "How does FAISS implement cosine similarity?",
            "What is ChromaDB used for?",
        ]
        for q in questions:
            print(f"Q: {q}")
            response = pipeline.query(q)
            print(f"A: {response.answer}")
            print(f"   Sources used: {len(response.sources)}")
            print()


if __name__ == "__main__":
    main()
