"""
LLM client factory + RAGChain — connects any provider to a retriever + prompt.
Supported providers: groq, gemini, ollama, cerebras, openai.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ingestion.document_loader import Document
from retrieval.retriever import DenseRetriever, HybridRetriever
from generation.prompt_templates import build_context_string, format_qa_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str: ...

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str: ...


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


class GroqClient(LLMClient):
    """Groq cloud inference — ultra-low latency LLM API."""

    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile") -> None:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq required: pip install groq")
        self._client = Groq(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content


class GeminiClient(LLMClient):
    """Google Gemini via google-generativeai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError(
                "google-generativeai required: pip install google-generativeai"
            )

    def complete(self, prompt: str, **kwargs: Any) -> str:
        resp = self._model.generate_content(prompt, **kwargs)
        return resp.text

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        # Convert OpenAI-style messages to Gemini history
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        chat = self._model.start_chat(history=history)
        resp = chat.send_message(messages[-1]["content"], **kwargs)
        return resp.text


class OllamaClient(LLMClient):
    """Local Ollama model server."""

    def __init__(
        self, model: str = "llama3.2", host: str = "http://localhost:11434"
    ) -> None:
        try:
            import ollama

            self._client = ollama.Client(host=host)
        except ImportError:
            raise ImportError("ollama required: pip install ollama")
        self._model = model

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self._client.chat(model=self._model, messages=messages, **kwargs)
        return resp["message"]["content"]


class CerebrasClient(LLMClient):
    """Cerebras cloud inference."""

    def __init__(self, api_key: str, model: str = "llama3.1-70b") -> None:
        try:
            from cerebras.cloud.sdk import Cerebras
        except ImportError:
            raise ImportError(
                "cerebras-cloud-sdk required: pip install cerebras-cloud-sdk"
            )
        self._client = Cerebras(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content


class OpenAIClient(LLMClient):
    """OpenAI-compatible client (also works with local vLLM/LM Studio)."""

    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", base_url: Optional[str] = None
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai required: pip install openai")
        self._client = OpenAI(
            api_key=api_key, **({"base_url": base_url} if base_url else {})
        )
        self._model = model

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_client(
    provider: str,
    api_key: str = "",
    model: str = "",
    **kwargs: Any,
) -> LLMClient:
    """
    Factory for LLM clients.

    Args:
        provider: One of "groq", "gemini", "ollama", "cerebras", "openai".
        api_key: Provider API key (not needed for ollama).
        model: Model name override.

    Returns:
        Configured LLMClient instance.
    """
    provider = provider.lower()
    defaults: Dict[str, Dict] = {
        "groq": {"model": "llama-3.1-70b-versatile"},
        "gemini": {"model": "gemini-1.5-flash"},
        "ollama": {"model": "llama3.2"},
        "cerebras": {"model": "llama3.1-70b"},
        "openai": {"model": "gpt-4o-mini"},
    }
    if provider not in defaults:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {list(defaults)}"
        )

    used_model = model or defaults[provider]["model"]

    if provider == "groq":
        return GroqClient(api_key=api_key, model=used_model)
    if provider == "gemini":
        return GeminiClient(api_key=api_key, model=used_model)
    if provider == "ollama":
        return OllamaClient(model=used_model, **kwargs)
    if provider == "cerebras":
        return CerebrasClient(api_key=api_key, model=used_model)
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=used_model, **kwargs)
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# RAGChain
# ---------------------------------------------------------------------------


@dataclass
class RAGResponse:
    answer: str
    sources: List[Document]
    scores: List[float]


class RAGChain:
    """
    End-to-end RAG chain: retrieve → format prompt → generate.

    Example:
        chain = RAGChain(retriever=retriever, llm=groq_client)
        resp = chain.query("What is FAISS?")
        print(resp.answer)
    """

    def __init__(
        self,
        retriever: DenseRetriever | HybridRetriever,
        llm: LLMClient,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._system_prompt = system_prompt or (
            "You are a helpful assistant. Use only the provided context to answer."
        )

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        generation_kwargs: Optional[Dict] = None,
    ) -> RAGResponse:
        k = top_k or self._top_k
        results = self._retriever.retrieve(question, top_k=k)
        docs = [r.document for r in results]
        scores = [r.score for r in results]

        prompt = format_qa_prompt(question, docs)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        answer = self._llm.chat(messages, **(generation_kwargs or {}))
        return RAGResponse(answer=answer, sources=docs, scores=scores)

    def query_with_history(
        self,
        question: str,
        history: List[Dict[str, str]],
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """Multi-turn RAG with conversation history."""
        k = top_k or self._top_k
        results = self._retriever.retrieve(question, top_k=k)
        docs = [r.document for r in results]
        context = build_context_string(docs)

        user_msg = f"Context:\n{context}\n\nQuestion: {question}"
        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_msg})
        answer = self._llm.chat(messages)
        return RAGResponse(
            answer=answer, sources=docs, scores=[r.score for r in results]
        )
