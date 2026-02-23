"""
Online RAG example — Pinecone + Groq / Gemini / Cerebras.
Set environment variables before running:

    export PINECONE_API_KEY="..."
    export LLM_PROVIDER="groq"       # groq | gemini | cerebras | openai
    export LLM_API_KEY="..."
    export LLM_MODEL=""              # leave empty for provider default

Run:  python examples/pinecone_rag_example.py
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.document_loader import Document
from pipelines.online_rag import OnlineRAGConfig, OnlineRAGPipeline
from generation.structured_output import ExtractedFacts, JSONModeExtractor
from generation.chain import create_llm_client

SAMPLE_DOCS = [
    Document(
        page_content=(
            "Pinecone is a fully managed vector database for machine learning applications. "
            "It supports serverless deployments and uses approximate nearest-neighbour search "
            "via the HNSW algorithm. Pinecone namespaces allow logical partitioning of vectors "
            "within a single index."
        ),
        metadata={"source": "pinecone-docs", "topic": "vector-db"},
    ),
    Document(
        page_content=(
            "Groq provides ultra-low latency LLM inference using its custom LPU Inference Engine. "
            "It is compatible with the OpenAI API surface, making it a drop-in replacement. "
            "Groq can process thousands of tokens per second for models like Llama 3."
        ),
        metadata={"source": "groq-docs", "topic": "llm-inference"},
    ),
    Document(
        page_content=(
            "Cerebras offers cloud inference on wafer-scale AI chips. The Cerebras CS-3 system "
            "uses the world's largest processor, enabling very fast inference for large language models."
        ),
        metadata={"source": "cerebras-docs", "topic": "llm-inference"},
    ),
]


def run_pipeline() -> None:
    pinecone_key = os.getenv("PINECONE_API_KEY", "")
    llm_provider = os.getenv("LLM_PROVIDER", "groq")
    llm_key = os.getenv("LLM_API_KEY", "")
    llm_model = os.getenv("LLM_MODEL", "")

    if not pinecone_key:
        print("PINECONE_API_KEY not set — skipping cloud test.")
        return
    if not llm_key and llm_provider != "ollama":
        print(
            f"LLM_API_KEY not set for provider '{llm_provider}' — skipping cloud test."
        )
        return

    cfg = OnlineRAGConfig(
        pinecone_api_key=pinecone_key,
        pinecone_index="boilerplate-rag-demo",
        llm_provider=llm_provider,
        llm_api_key=llm_key,
        llm_model=llm_model,
        top_k=3,
        use_reranker=False,
    )
    pipeline = OnlineRAGPipeline(config=cfg)

    # --- ingest sample docs ---
    n = pipeline.ingest_documents(SAMPLE_DOCS)
    print(f"\nUpserted {n} chunks to Pinecone.\n")

    # --- basic QA ---
    questions = [
        "What makes Pinecone suitable for ML applications?",
        "Why is Groq inference so fast?",
    ]
    for q in questions:
        print(f"Q: {q}")
        resp = pipeline.query(q)
        print(f"A: {resp.answer}\n")

    # --- multi-turn conversation ---
    print("--- Multi-turn conversation ---")
    history = []
    turns = [
        "What is Pinecone?",
        "Can you compare it to FAISS?",
    ]
    for turn in turns:
        resp = pipeline.query_with_history(turn, history=history)
        print(f"User: {turn}")
        print(f"AI:   {resp.answer}\n")
        history.append({"role": "user", "content": turn})
        history.append({"role": "assistant", "content": resp.answer})

    # --- structured output demo ---
    print("--- Structured output ---")
    llm = create_llm_client(provider=llm_provider, api_key=llm_key, model=llm_model)
    extractor = JSONModeExtractor(llm)
    context = "\n".join(d.page_content for d in SAMPLE_DOCS[:2])
    facts = extractor.extract(ExtractedFacts, "Extract key technical facts", context)
    print(f"Extracted {len(facts.facts)} facts:")
    for f in facts.facts:
        print(f"  • {f}")


if __name__ == "__main__":
    run_pipeline()
