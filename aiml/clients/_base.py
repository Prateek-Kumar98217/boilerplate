"""
Abstract base class wiring together key rotation, memory, and prompt templates
into a unified inference client interface.

Every concrete client (Gemini, Groq, Cerebras) inherits from BaseClient.

Public interface:
  client.chat(message)                  → str
  client.complete(prompt)               → str
  async client.achat(message)           → str
  async client.acomplete(prompt)        → str
  async client.astream(message)         → AsyncGenerator[str]
  client.memory                         → MemoryManager
  client.template                       → PromptTemplate (active)
  client.set_template(tmpl)
  client.set_system(system_str)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from _key_rotator import AllKeysExhaustedError, KeyRotator
from _memory import MemoryManager, SummariserFn
from _prompt import PromptTemplate, Templates

logger = logging.getLogger(__name__)

# Maximum retries on rate-limit before propagating
_MAX_RL_RETRIES = 4


class BaseClient(ABC):
    """
    Abstract inference client with built-in:
      • API key rotation (rate-limit-aware)
      • Automatic short + long term memory with summarisation
      • Customisable prompt templates

    Subclasses must implement:
      _raw_chat(messages, model, **kw)  →  str
      _raw_stream(messages, model, **kw) → AsyncGenerator[str]
      default_model  (property or class attr)
    """

    def __init__(
        self,
        rotator: KeyRotator,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        template: Optional[PromptTemplate] = None,
        memory_max_turns: int = 20,
        memory_max_chars: int = 8000,
        enable_memory: bool = True,
    ) -> None:
        self._rotator = rotator
        self._model = model or self.default_model
        self._template: PromptTemplate = template or Templates.chat
        self._enable_memory = enable_memory
        self._system_prompt = system_prompt or (
            self._template.system or "You are a helpful AI assistant."
        )
        self._memory = MemoryManager(
            max_short_turns=memory_max_turns,
            max_short_chars=memory_max_chars,
            summariser=self._make_summariser(),
            system_prompt=self._system_prompt,
        )

    # ------------------------------------------------------------------
    # Overrideable properties
    # ------------------------------------------------------------------

    @property
    def default_model(self) -> str:  # subclass should override
        return "unknown"

    # ------------------------------------------------------------------
    # Public: synchronous
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        system_override: Optional[str] = None,
        extra_system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Send a user message, update memory, return the assistant reply.

        Args:
            message:         User message text.
            system_override: Replace system prompt for this turn only.
            extra_system:    Append extra instructions this turn only.
            model:           Override the default model for this call.
            **kwargs:        Passed through to the underlying API.
        """
        return asyncio.run(
            self.achat(
                message,
                system_override=system_override,
                extra_system=extra_system,
                model=model,
                **kwargs,
            )
        )

    def complete(self, prompt: str, model: Optional[str] = None, **kwargs: Any) -> str:
        """
        Stateless completion — does NOT update memory.

        Args:
            prompt: Raw text prompt.
            model:  Override model for this call.
        """
        return asyncio.run(self.acomplete(prompt, model=model, **kwargs))

    # ------------------------------------------------------------------
    # Public: async
    # ------------------------------------------------------------------

    async def achat(
        self,
        message: str,
        system_override: Optional[str] = None,
        extra_system: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        if self._enable_memory:
            self._memory.add("user", message)

        messages = self._build_messages(
            system_override=system_override,
            extra_system=extra_system,
        )
        reply = await self._call_with_retry(messages, model=model, **kwargs)

        if self._enable_memory:
            self._memory.add("assistant", reply)
        return reply

    async def acomplete(
        self, prompt: str, model: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Stateless completion (no memory update)."""
        messages = [{"role": "user", "content": prompt}]
        return await self._call_with_retry(messages, model=model, **kwargs)

    async def astream(
        self,
        message: str,
        system_override: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens.  Memory is updated at the end of the stream.

        Yields:
            Partial token strings.
        """
        if self._enable_memory:
            self._memory.add("user", message)

        messages = self._build_messages(system_override=system_override)
        key = await self._rotator.acquire()
        full_reply = ""
        try:
            async for token in self._raw_stream(
                messages=messages, model=model or self._model, key=key, **kwargs
            ):
                full_reply += token
                yield token
        except Exception as exc:
            _handle_rate_limit(exc, self._rotator, key)
            raise
        finally:
            self._rotator.release(key)

        if self._enable_memory:
            self._memory.add("assistant", full_reply)

    # ------------------------------------------------------------------
    # Template & memory configuration
    # ------------------------------------------------------------------

    def set_template(self, template: PromptTemplate) -> None:
        """Replace the active prompt template."""
        self._template = template

    def set_system(self, system_prompt: str) -> None:
        """Update the system prompt (affects memory context on next call)."""
        self._system_prompt = system_prompt
        self._memory._system_prompt = system_prompt

    def set_model(self, model: str) -> None:
        """Change the default model."""
        self._model = model

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def template(self) -> PromptTemplate:
        return self._template

    def reset_memory(self) -> None:
        self._memory.clear()

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    async def _raw_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str,
        **kwargs: Any,
    ) -> str: ...

    async def _raw_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Override to support streaming. Default falls back to full chat."""
        full = await self._raw_chat(messages=messages, model=model, key=key, **kwargs)
        yield full

    # ------------------------------------------------------------------
    # Internal machinery
    # ------------------------------------------------------------------

    async def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        used_model = model or self._model
        for attempt in range(_MAX_RL_RETRIES):
            key = await self._rotator.acquire()
            try:
                reply = await self._raw_chat(
                    messages=messages, model=used_model, key=key, **kwargs
                )
                self._rotator.release(key)
                return reply
            except Exception as exc:
                retry_after = _handle_rate_limit(exc, self._rotator, key)
                self._rotator.release(key)
                if retry_after is None:
                    raise  # not a rate-limit error → propagate immediately
                wait = retry_after or (2**attempt * 5)
                logger.warning(
                    "%s: rate limited on attempt %d, waiting %.1fs",
                    self.__class__.__name__,
                    attempt + 1,
                    wait,
                )
                await asyncio.sleep(wait)

        raise AllKeysExhaustedError("All retries exhausted due to rate limiting.")

    def _build_messages(
        self,
        system_override: Optional[str] = None,
        extra_system: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        if self._enable_memory:
            return self._memory.get_context(
                system_override=system_override,
                extra_system=extra_system,
            )
        # Memory disabled: bare system + user messages
        msgs = []
        sys_content = system_override or self._system_prompt
        if extra_system:
            sys_content = (sys_content + "\n\n" + extra_system).strip()
        if sys_content:
            msgs.append({"role": "system", "content": sys_content})
        return msgs

    def _make_summariser(self) -> SummariserFn:
        """Return a summariser callback that calls this client's own API."""

        def _summarise(prompt: str) -> str:
            return asyncio.run(
                self._call_with_retry([{"role": "user", "content": prompt}])
            )

        return _summarise


# ---------------------------------------------------------------------------
# Shared rate-limit detection helper
# ---------------------------------------------------------------------------


def _handle_rate_limit(
    exc: Exception,
    rotator: KeyRotator,
    key: str,
) -> Optional[float]:
    """
    If ``exc`` is a rate-limit error, mark the key and return the suggested
    wait time (seconds).  Returns None if it's a different kind of error.
    """
    msg = str(exc).lower()
    is_rl = (
        "rate limit" in msg
        or "rate_limit" in msg
        or "429" in msg
        or "too many requests" in msg
        or "quota" in msg
    )
    if not is_rl:
        return None

    # Try to extract Retry-After
    retry_after: Optional[float] = None
    if hasattr(exc, "response"):
        hdrs = getattr(exc.response, "headers", {}) or {}
        ra = hdrs.get("retry-after") or hdrs.get("Retry-After")
        if ra:
            try:
                retry_after = float(ra)
            except ValueError:
                pass

    rotator.mark_rate_limited(key, retry_after_s=retry_after)
    return retry_after
