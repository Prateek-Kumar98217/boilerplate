"""
Automatic memory management for inference clients.

Architecture:
  ┌──────────────────────────────────────────┐
  │              MemoryManager               │
  │                                          │
  │  short_term: List[Message]   (verbatim)  │
  │  long_term:  str             (summary)   │
  │                                          │
  │  When short-term exceeds MAX_TURNS,      │
  │  the oldest half is summarised via LLM   │
  │  and merged into long_term.              │
  └──────────────────────────────────────────┘

The manager exposes get_context() which returns:
  [system_msg?, long_term_summary_msg?, ...short_term_msgs]

ready to be passed directly as the `messages` list to any OpenAI-compatible API.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# Summariser protocol — any callable (llm_fn) that takes a str → str
# ---------------------------------------------------------------------------

SummariserFn = Callable[[str], str]

_SUMMARISE_PROMPT = """\
You are a memory compressor. Produce a dense, factual summary of the \
conversation excerpt below. Preserve all key facts, decisions, values, \
names, and numbers. Be concise.

{previous_summary}

New conversation to merge:
{conversation}

Updated summary:"""


def make_default_summariser(llm_complete_fn: SummariserFn) -> SummariserFn:
    """Wrap a raw complete() function with the compression prompt."""

    def summarise(text: str) -> str:
        return llm_complete_fn(text)

    return summarise


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class MemoryManager:
    """
    Combined short-term + long-term memory with automatic summarisation.

    Args:
        max_short_turns:  Max number of (user+assistant) pairs kept verbatim.
        max_short_chars:  Soft char limit on the short-term window.
        summariser:       Callable(prompt: str) → str used to compress memory.
                          If None, overflow is silently dropped (not summarised).
        system_prompt:    Fixed system message injected at the start of context.

    Example:
        mem = MemoryManager(max_short_turns=10, summariser=groq_complete)
        mem.add("user", "Hello!")
        mem.add("assistant", "Hi! How can I help?")
        messages = mem.get_context()   # pass to API
    """

    def __init__(
        self,
        max_short_turns: int = 20,
        max_short_chars: int = 8000,
        summariser: Optional[SummariserFn] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._max_turns = max_short_turns
        self._max_chars = max_short_chars
        self._summariser = summariser
        self._system_prompt = system_prompt

        self._short: List[Message] = []
        self._long_summary: str = ""
        self._turn_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Append a message and trigger compression if limits are hit."""
        self._short.append(Message(role=role, content=content, metadata=metadata or {}))
        if role in ("user", "assistant"):
            self._turn_count += 1

        if self._should_compress():
            self._compress()

    def add_message(self, msg: Dict[str, str]) -> None:
        """Convenience: add from a dict with 'role' and 'content' keys."""
        self.add(msg["role"], msg["content"])

    def get_context(
        self,
        system_override: Optional[str] = None,
        extra_system: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build the messages list to pass to the LLM.

        Order:
          1. System message (override > default > none)
          2. Long-term summary injection (if non-empty)
          3. Short-term verbatim messages

        Args:
            system_override: Replace the configured system prompt entirely.
            extra_system:    Append extra instructions to the system message.
        """
        messages = []

        # System
        sys_content = system_override or self._system_prompt or ""
        if extra_system:
            sys_content = (sys_content + "\n\n" + extra_system).strip()
        if sys_content:
            messages.append({"role": "system", "content": sys_content})

        # Long-term summary
        if self._long_summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "[Conversation summary — earlier context]\n"
                        + self._long_summary
                    ),
                }
            )

        # Short-term verbatim
        messages.extend(m.to_dict() for m in self._short)
        return messages

    def clear(self) -> None:
        """Wipe all memory (short + long)."""
        self._short = []
        self._long_summary = ""
        self._turn_count = 0

    def clear_short_term(self) -> None:
        """Drop short-term messages, keep the long-term summary."""
        self._short = []
        self._turn_count = 0

    def inject_long_term(self, summary: str) -> None:
        """Manually seed the long-term summary (e.g. loaded from storage)."""
        self._long_summary = summary

    @property
    def long_term_summary(self) -> str:
        return self._long_summary

    @property
    def short_term_messages(self) -> List[Dict[str, str]]:
        return [m.to_dict() for m in self._short]

    @property
    def total_chars(self) -> int:
        return sum(len(m.content) for m in self._short)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def snapshot(self) -> Dict[str, Any]:
        """Serialisable snapshot for persistence."""
        return {
            "long_summary": self._long_summary,
            "short_term": [m.to_dict() for m in self._short],
            "turn_count": self._turn_count,
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore from a snapshot dict."""
        self._long_summary = snapshot.get("long_summary", "")
        self._turn_count = snapshot.get("turn_count", 0)
        self._short = [
            Message(role=m["role"], content=m["content"])
            for m in snapshot.get("short_term", [])
        ]

    # ------------------------------------------------------------------
    # Internal compression
    # ------------------------------------------------------------------

    def _should_compress(self) -> bool:
        return self._turn_count > self._max_turns or self.total_chars > self._max_chars

    def _compress(self) -> None:
        """
        Move the oldest half of short-term messages into the long-term summary.
        """
        if not self._short:
            return

        half = max(1, len(self._short) // 2)
        to_compress = self._short[:half]
        self._short = self._short[half:]
        self._turn_count = max(0, self._turn_count - half)

        # Build readable transcript
        transcript = "\n".join(
            f"{m.role.capitalize()}: {m.content}" for m in to_compress
        )

        if self._summariser is None:
            # No summariser: just prepend a plain excerpt
            self._long_summary = (
                self._long_summary + "\n\n[Earlier messages omitted for brevity]"
            ).strip()
            logger.debug(
                "MemoryManager: no summariser configured — dropped %d messages",
                len(to_compress),
            )
            return

        try:
            prev = (
                f"Current summary:\n{self._long_summary}\n\n"
                if self._long_summary
                else ""
            )
            prompt = _SUMMARISE_PROMPT.format(
                previous_summary=prev,
                conversation=transcript,
            )
            new_summary = self._summariser(prompt)
            self._long_summary = new_summary.strip()
            logger.debug(
                "MemoryManager: compressed %d messages into long-term summary (%d chars)",
                len(to_compress),
                len(self._long_summary),
            )
        except Exception as exc:
            logger.warning(
                "MemoryManager: summarisation failed (%s); dropping old messages", exc
            )
            self._long_summary = (
                self._long_summary + "\n\n[earlier context condensed]"
            ).strip()
