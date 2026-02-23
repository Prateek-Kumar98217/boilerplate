"""
Agent memory modules — buffer, summary, and vector-based memory.

BufferMemory       — sliding window of recent messages (token-limited)
SummaryMemory      — LLM-summarised long-term memory
VectorMemory       — FAISS-backed episodic memory with semantic retrieval
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data model
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    role: str
    content: str
    turn: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseMemory(ABC):
    @abstractmethod
    def add(self, role: str, content: str) -> None: ...

    @abstractmethod
    def get_context(self, query: str = "") -> List[Dict[str, str]]: ...

    @abstractmethod
    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# 1. Buffer memory — keeps last N tokens
# ---------------------------------------------------------------------------


class BufferMemory(BaseMemory):
    """
    Simple sliding-window memory capped at max_tokens (rough char-based estimate).

    Example:
        mem = BufferMemory(max_tokens=2048)
        mem.add("user", "Hello!")
        context = mem.get_context()
    """

    def __init__(self, max_tokens: int = 4096, chars_per_token: float = 4.0) -> None:
        self._max_chars = int(max_tokens * chars_per_token)
        self._entries: List[MemoryEntry] = []
        self._turn = 0

    def add(self, role: str, content: str) -> None:
        self._entries.append(MemoryEntry(role=role, content=content, turn=self._turn))
        self._turn += 1
        self._trim()

    def get_context(self, query: str = "") -> List[Dict[str, str]]:
        return [{"role": e.role, "content": e.content} for e in self._entries]

    def clear(self) -> None:
        self._entries = []
        self._turn = 0

    def _trim(self) -> None:
        while self._total_chars() > self._max_chars and len(self._entries) > 1:
            self._entries.pop(0)

    def _total_chars(self) -> int:
        return sum(len(e.content) for e in self._entries)


# ---------------------------------------------------------------------------
# 2. Summary memory — LLM-compressed long-term context
# ---------------------------------------------------------------------------


class SummaryMemory(BaseMemory):
    """
    Maintains a rolling LLM-generated summary of the conversation.
    Combines the summary with the most recent messages for each context window.

    Example:
        mem = SummaryMemory(llm=groq_client, buffer_turns=6)
        mem.add("user", "Tell me about transformers.")
        mem.add("assistant", "Transformers are...")
    """

    SUMMARY_PROMPT = (
        "Progressively summarise the conversation below into a concise paragraph. "
        "Preserve key facts, decisions, and values mentioned.\n\n"
        "Current summary:\n{summary}\n\nNew messages:\n{new}\n\nUpdated summary:"
    )

    def __init__(
        self,
        llm: Any,
        buffer_turns: int = 6,
        max_summary_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._buffer_turns = buffer_turns
        self._max_summary_tokens = max_summary_tokens
        self._summary: str = ""
        self._recent: List[MemoryEntry] = []
        self._unsummarised: List[MemoryEntry] = []
        self._turn = 0

    def add(self, role: str, content: str) -> None:
        entry = MemoryEntry(role=role, content=content, turn=self._turn)
        self._turn += 1
        self._recent.append(entry)
        self._unsummarised.append(entry)

        if len(self._recent) > self._buffer_turns:
            self._recent.pop(0)

        # Compress every buffer_turns turns
        if len(self._unsummarised) >= self._buffer_turns:
            self._compress()

    def _compress(self) -> None:
        new_text = "\n".join(
            f"{e.role.capitalize()}: {e.content}" for e in self._unsummarised
        )
        prompt = self.SUMMARY_PROMPT.format(summary=self._summary, new=new_text)
        try:
            self._summary = self._llm.complete(prompt)
        except Exception as exc:
            logger.warning("SummaryMemory compression failed: %s", exc)
        self._unsummarised = []

    def get_context(self, query: str = "") -> List[Dict[str, str]]:
        messages = []
        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"[Conversation summary so far]\n{self._summary}",
                }
            )
        messages.extend({"role": e.role, "content": e.content} for e in self._recent)
        return messages

    def clear(self) -> None:
        self._summary = ""
        self._recent = []
        self._unsummarised = []
        self._turn = 0


# ---------------------------------------------------------------------------
# 3. Vector memory — FAISS episodic retrieval
# ---------------------------------------------------------------------------


class VectorMemory(BaseMemory):
    """
    Stores every message as an embedding in a FAISS index.
    On retrieval, returns the most semantically relevant past messages.

    Example:
        mem = VectorMemory(embedding_model="all-MiniLM-L6-v2", top_k=4)
        mem.add("user", "What is quantum entanglement?")
        context = mem.get_context("entanglement and teleportation")
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            import faiss
        except ImportError:
            raise ImportError(
                "sentence-transformers and faiss required: "
                "pip install sentence-transformers faiss-cpu"
            )
        import numpy as np
        import faiss

        self._model = SentenceTransformer(embedding_model, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatIP(self._dim)
        self._entries: List[MemoryEntry] = []
        self._top_k = top_k
        self._np = np
        self._faiss = faiss
        self._turn = 0

    def add(self, role: str, content: str) -> None:
        entry = MemoryEntry(role=role, content=content, turn=self._turn)
        self._entries.append(entry)
        self._turn += 1

        vec = self._model.encode([content], normalize_embeddings=True)
        self._index.add(self._np.array(vec, dtype="float32"))

    def get_context(self, query: str = "") -> List[Dict[str, str]]:
        if self._index.ntotal == 0:
            return []
        if not query:
            # Return all recent if no query
            return [
                {"role": e.role, "content": e.content}
                for e in self._entries[-self._top_k :]
            ]

        qvec = self._model.encode([query], normalize_embeddings=True)
        k = min(self._top_k, self._index.ntotal)
        _, indices = self._index.search(self._np.array(qvec, dtype="float32"), k)
        retrieved = sorted(
            [self._entries[i] for i in indices[0] if i >= 0], key=lambda x: x.turn
        )
        return [{"role": e.role, "content": e.content} for e in retrieved]

    def clear(self) -> None:
        import faiss  # lazy re-import for reset

        self._index = faiss.IndexFlatIP(self._dim)
        self._entries = []
        self._turn = 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_memory(memory_type: str = "buffer", **kwargs: Any) -> BaseMemory:
    """
    Factory to create a memory instance by name.

    Args:
        memory_type: "buffer" | "summary" | "vector"
        **kwargs: Forwarded to the memory class constructor.
    """
    if memory_type == "buffer":
        return BufferMemory(**kwargs)
    if memory_type == "summary":
        return SummaryMemory(**kwargs)
    if memory_type == "vector":
        return VectorMemory(**kwargs)
    raise ValueError(
        f"Unknown memory type '{memory_type}'. Choose: buffer|summary|vector"
    )
