"""
Cerebras inference client.

Install: pip install cerebras-cloud-sdk

Cerebras provides ultra-fast LLM inference via an OpenAI-compatible API.
This client supports:
  • Text chat / completion              — CerebrasClient.chat() / complete()
  • Streaming                           — CerebrasClient.astream()
  • Multiple models with easy switching — client.set_model(...)

Available models:
  • llama3.1-8b
  • llama3.1-70b     (default)
  • llama3.3-70b
  • qwen-3-32b
  • deepseek-r1-distill-llama-70b

Key rotation:
  Pass multiple API keys via CEREBRAS_API_KEYS=key1,key2,key3 in .env
  or supply a list directly to the constructor.

Memory:
  Automatic short-term + long-term summarisation enabled by default.
  Access via client.memory.  Disable with enable_memory=False.

Prompt templates:
  client.set_template(Templates.chain_of_thought)
  client.set_system("You are a reasoning expert.")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from _base import BaseClient, _handle_rate_limit
from _key_rotator import KeyRotator
from _prompt import PromptTemplate, Templates

logger = logging.getLogger(__name__)

# Available Cerebras models for quick reference
CEREBRAS_MODELS = {
    "llama3.1-8b": "llama3.1-8b",
    "llama3.1-70b": "llama3.1-70b",
    "llama3.3-70b": "llama3.3-70b",
    "qwen-3-32b": "qwen-3-32b",
    "deepseek-r1": "deepseek-r1-distill-llama-70b",
}


class CerebrasClient(BaseClient):
    """
    Cerebras LLM client with key rotation, memory, and prompt templates.

    Args:
        api_keys:         One or more Cerebras API keys (rotated automatically).
        model:            Default model ID (default: llama3.1-70b).
        rpm_limit:        Requests-per-minute per key.
        system_prompt:    Global system instruction.
        template:         Default PromptTemplate.
        memory_max_turns: Max verbatim turns in short-term memory.
        memory_max_chars: Char limit for short-term window.
        enable_memory:    Toggle memory on/off.
        default_params:   Extra generation kwargs sent on every request
                          (temperature, max_tokens, top_p, seed, etc.).

    Example:
        client = CerebrasClient.from_env()
        reply = client.chat("What is sparsity in neural networks?")

        # Streaming
        async for token in client.astream("Explain backpropagation."):
            print(token, end="", flush=True)

        # Switch model per call
        reply = client.chat("Draft code.", model="llama3.3-70b")
    """

    def __init__(
        self,
        api_keys: List[str],
        model: str = "llama3.1-70b",
        rpm_limit: int = 30,
        system_prompt: Optional[str] = None,
        template: Optional[PromptTemplate] = None,
        memory_max_turns: int = 20,
        memory_max_chars: int = 8000,
        enable_memory: bool = True,
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        rotator = KeyRotator(keys=api_keys, rpm_limit=rpm_limit)
        super().__init__(
            rotator=rotator,
            model=model,
            system_prompt=system_prompt,
            template=template or Templates.chat,
            memory_max_turns=memory_max_turns,
            memory_max_chars=memory_max_chars,
            enable_memory=enable_memory,
        )
        self._default_params: Dict[str, Any] = default_params or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, **kwargs: Any) -> "CerebrasClient":
        """Construct from environment / .env file."""
        from config import get_settings

        cfg = get_settings().cerebras
        keys = cfg.api_keys or [os.environ.get("CEREBRAS_API_KEY", "")]
        return cls(
            api_keys=[k for k in keys if k],
            model=cfg.default_model,
            rpm_limit=cfg.rpm_limit,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Abstract implementation
    # ------------------------------------------------------------------

    @property
    def default_model(self) -> str:
        return "llama3.1-70b"

    async def _raw_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str,
        **kwargs: Any,
    ) -> str:
        cerebras = _import_cerebras()
        client = cerebras.Cerebras(api_key=key)
        params = {**self._default_params, **kwargs}
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            **params,
        )
        return _extract_content(resp)

    async def _raw_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        cerebras = _import_cerebras()
        client = cerebras.Cerebras(api_key=key)
        params = {**self._default_params, **kwargs}

        def _iter():
            with client.chat.completions.stream(
                model=model,
                messages=messages,
                **params,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield delta.content

        for token in await asyncio.to_thread(lambda: list(_iter())):
            yield token

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return the known Cerebras model IDs."""
        return list(CEREBRAS_MODELS.values())

    def key_status(self) -> List[dict]:
        """Return current rate-limit status for all API keys."""
        return self._rotator.status()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_cerebras() -> Any:
    try:
        import cerebras.cloud.sdk as cerebras  # type: ignore

        return cerebras
    except ImportError:
        raise ImportError("cerebras-cloud-sdk required: pip install cerebras-cloud-sdk")


def _extract_content(resp: Any) -> str:
    """Extract text from a Cerebras (OpenAI-compat) response object."""
    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError) as exc:
        raise ValueError(f"Unexpected Cerebras response format: {resp}") from exc
