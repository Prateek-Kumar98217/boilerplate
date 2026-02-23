"""
Structured output extraction from RAG context using Pydantic + instructor or JSON mode.
Install: pip install instructor openai
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from generation.chain import LLMClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Example Pydantic schemas
# ---------------------------------------------------------------------------


class CitedAnswer(BaseModel):
    """A question answer with supporting citations."""

    answer: str = Field(description="Direct answer to the question")
    citations: List[str] = Field(
        default_factory=list, description="Source passages used"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence 0-1")


class ExtractedFacts(BaseModel):
    """Discrete facts extracted from context."""

    facts: List[str] = Field(description="Individual factual statements")
    entities: Dict[str, List[str]] = Field(
        default_factory=dict, description="Named entities keyed by type"
    )
    summary: str = Field(default="", description="One-sentence summary")


class SearchResult(BaseModel):
    """Structured search result."""

    title: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    url: Optional[str] = None


class ClassificationResult(BaseModel):
    """Document or query classification."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Instructor-based extractor (preferred — type-safe)
# ---------------------------------------------------------------------------


class StructuredExtractor:
    """
    Uses `instructor` to extract structured Pydantic models from any LLM.
    Supports groq, openai, and ollama backends.

    Example:
        extractor = StructuredExtractor.from_groq(api_key, "llama-3.1-8b-instant")
        result = extractor.extract(CitedAnswer, question, context)
    """

    def __init__(self, client: Any, mode: str = "json") -> None:
        """
        Args:
            client: An instructor-patched or raw OpenAI-compatible client.
            mode: "json" for JSON mode, "tools" for function calling.
        """
        self._client = client
        self._mode = mode

    @classmethod
    def from_groq(
        cls, api_key: str, model: str = "llama-3.1-8b-instant"
    ) -> "StructuredExtractor":
        try:
            import instructor
            from groq import Groq
        except ImportError:
            raise ImportError("Install: pip install instructor groq")
        client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)
        inst = cls.__new__(cls)
        inst._raw_client = client
        inst._model = model
        inst._use_instructor = True
        return inst

    @classmethod
    def from_openai(
        cls, api_key: str, model: str = "gpt-4o-mini"
    ) -> "StructuredExtractor":
        try:
            import instructor
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install: pip install instructor openai")
        client = instructor.from_openai(OpenAI(api_key=api_key))
        inst = cls.__new__(cls)
        inst._raw_client = client
        inst._model = model
        inst._use_instructor = True
        return inst

    def extract(
        self,
        schema: Type[T],
        question: str,
        context: str,
        max_retries: int = 2,
    ) -> T:
        if getattr(self, "_use_instructor", False):
            return self._raw_client.chat.completions.create(
                model=self._model,
                response_model=schema,
                max_retries=max_retries,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract structured information from the context accurately.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nTask: {question}",
                    },
                ],
            )
        raise RuntimeError(
            "Call from_groq() or from_openai() to initialise instructor."
        )


# ---------------------------------------------------------------------------
# JSON-mode extractor (works with any LLMClient, no instructor dependency)
# ---------------------------------------------------------------------------


class JSONModeExtractor:
    """
    Asks any LLMClient to return JSON matching a schema.
    Falls back to regex extraction if the model wraps JSON in markdown blocks.

    Example:
        extractor = JSONModeExtractor(groq_llm_client)
        result = extractor.extract(CitedAnswer, question, context)
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def extract(
        self,
        schema: Type[T],
        question: str,
        context: str,
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Task: {question}\n\n"
            f"Return ONLY valid JSON matching this schema (no explanation):\n{schema_json}"
        )
        raw = self._llm.complete(prompt)
        return self._parse(schema, raw)

    # ------------------------------------------------------------------
    def _parse(self, schema: Type[T], raw: str) -> T:
        # Strip markdown code fences if present
        raw = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()
        data = json.loads(raw)
        return schema(**data)
