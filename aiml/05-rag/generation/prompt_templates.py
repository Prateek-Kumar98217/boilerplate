"""
RAG prompt templates — Jinja2-compatible f-string templates for context-grounded generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ingestion.document_loader import Document


# ---------------------------------------------------------------------------
# Template strings
# ---------------------------------------------------------------------------

RAG_QA_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

RAG_CITATION_TEMPLATE = """\
You are a research assistant. Answer the question using the provided context.
For every factual claim, cite the source using [Source N] notation.
If the context lacks the answer, say "I don't know."

Context:
{context}

Question: {question}
Answer (with citations):"""

RAG_CONVERSATIONAL_TEMPLATE = """\
You are a conversational assistant with access to a knowledge base.
Use the context below to answer, while keeping a natural, friendly tone.
If the context doesn't cover the topic, say so politely.

Previous conversation:
{history}

Context:
{context}

User: {question}
Assistant:"""

RAG_STRUCTURED_OUTPUT_TEMPLATE = """\
You are a precise information extractor.
Using the context below, extract the requested information and return it as valid JSON.
Do not include fields that are not supported by the context.

Context:
{context}

Question / extraction request: {question}

Return strictly valid JSON matching this schema:
{schema}

JSON response:"""

HYPOTHETICAL_DOCUMENT_TEMPLATE = """\
Write a short passage (2-3 sentences) that would directly answer the following question.
Do NOT use any external knowledge beyond what is needed to answer.

Question: {question}
Passage:"""


# ---------------------------------------------------------------------------
# Helper dataclass
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate:
    template: str
    name: str = "custom"

    def format(self, **kwargs: str) -> str:
        return self.template.format(**kwargs)


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def build_context_string(
    docs: List[Document],
    separator: str = "\n\n---\n\n",
    max_chars_per_doc: int = 2000,
    add_source_numbers: bool = True,
) -> str:
    """Concatenate retrieved documents into a single context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        text = doc.page_content[:max_chars_per_doc]
        if add_source_numbers:
            parts.append(f"[Source {i}]\n{text}")
        else:
            parts.append(text)
    return separator.join(parts)


def format_qa_prompt(question: str, docs: List[Document]) -> str:
    context = build_context_string(docs)
    return RAG_QA_TEMPLATE.format(context=context, question=question)


def format_citation_prompt(question: str, docs: List[Document]) -> str:
    context = build_context_string(docs)
    return RAG_CITATION_TEMPLATE.format(context=context, question=question)


def format_conversational_prompt(
    question: str,
    docs: List[Document],
    history: Optional[str] = None,
) -> str:
    context = build_context_string(docs, add_source_numbers=False)
    hist = history or "(none)"
    return RAG_CONVERSATIONAL_TEMPLATE.format(
        context=context, question=question, history=hist
    )


def format_structured_output_prompt(
    question: str, docs: List[Document], schema: str
) -> str:
    context = build_context_string(docs)
    return RAG_STRUCTURED_OUTPUT_TEMPLATE.format(
        context=context, question=question, schema=schema
    )
