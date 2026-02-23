"""
Groq inference client.

Install: pip install groq

Groq provides two distinct input/output modes:
  1. LLM mode  — fast text chat/completion via GroqCloud
  2. Audio mode — Whisper speech-to-text transcription and translation

This client handles both modes cleanly and lets you configure each
independently (different models, params).

LLM models (examples):
  • llama-3.3-70b-versatile  (default)
  • llama-3.1-70b-versatile
  • llama-3.1-8b-instant
  • mixtral-8x7b-32768
  • gemma2-9b-it

Audio / Whisper models:
  • whisper-large-v3-turbo    (default, fastest)
  • whisper-large-v3          (most accurate)
  • distil-whisper-large-v3-en

Key rotation:
  GROQ_API_KEYS=key1,key2,key3  (or pass api_keys list to constructor)

Memory:
  Active for LLM conversations; not applicable to audio tasks.

Prompt templates:
  client.set_template(Templates.code_gen)
  client.set_system("You are a Python expert.")
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
from enum import Enum
from typing import Any, AsyncGenerator, BinaryIO, Dict, List, Optional, Union

from _base import BaseClient, _handle_rate_limit
from _key_rotator import KeyRotator
from _prompt import PromptTemplate, Templates

logger = logging.getLogger(__name__)

AudioInput = Union[str, pathlib.Path, bytes, BinaryIO]

# Groq supported audio formats
AUDIO_FORMATS = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}


class GroqMode(str, Enum):
    """Operating mode for GroqClient."""

    LLM = "llm"
    AUDIO = "audio"


class GroqClient(BaseClient):
    """
    Groq client supporting LLM chat and Whisper audio — with key rotation,
    memory, and prompt templates.

    Args:
        api_keys:           One or more Groq API keys.
        model:              Default LLM model.
        whisper_model:      Default Whisper model for audio tasks.
        rpm_limit:          Requests-per-minute per key.
        tpm_limit:          Tokens-per-minute per key (LLM only).
        system_prompt:      Global system instruction.
        template:           Default PromptTemplate.
        memory_max_turns:   Max verbatim turns.
        memory_max_chars:   Char budget for short-term window.
        enable_memory:      Toggle memory.
        default_llm_params: Extra kwargs for every LLM call (temperature, etc.).

    Example:
        client = GroqClient.from_env()

        # LLM chat
        reply = client.chat("Explain mixture-of-experts in two sentences.")

        # Streaming
        async for token in client.astream("Write a haiku about GPUs."):
            print(token, end="", flush=True)

        # Audio transcription
        text = client.transcribe("lecture.mp3")

        # Audio translation → English
        text = client.translate("german_speech.m4a")

        # Per-call model override
        reply = client.chat("Quick answer.", model="llama-3.1-8b-instant")
    """

    def __init__(
        self,
        api_keys: List[str],
        model: str = "llama-3.3-70b-versatile",
        whisper_model: str = "whisper-large-v3-turbo",
        rpm_limit: int = 30,
        tpm_limit: int = 131_072,
        system_prompt: Optional[str] = None,
        template: Optional[PromptTemplate] = None,
        memory_max_turns: int = 20,
        memory_max_chars: int = 8000,
        enable_memory: bool = True,
        default_llm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        rotator = KeyRotator(
            keys=api_keys,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
        )
        super().__init__(
            rotator=rotator,
            model=model,
            system_prompt=system_prompt,
            template=template or Templates.chat,
            memory_max_turns=memory_max_turns,
            memory_max_chars=memory_max_chars,
            enable_memory=enable_memory,
        )
        self._whisper_model = whisper_model
        self._default_llm_params: Dict[str, Any] = default_llm_params or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, **kwargs: Any) -> "GroqClient":
        from config import get_settings

        cfg = get_settings().groq
        keys = cfg.api_keys or [os.environ.get("GROQ_API_KEY", "")]
        return cls(
            api_keys=[k for k in keys if k],
            model=cfg.default_model,
            whisper_model=cfg.whisper_model,
            rpm_limit=cfg.rpm_limit,
            tpm_limit=cfg.tpm_limit,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # LLM — required abstract methods
    # ------------------------------------------------------------------

    @property
    def default_model(self) -> str:
        return "llama-3.3-70b-versatile"

    async def _raw_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str,
        **kwargs: Any,
    ) -> str:
        groq = _import_groq()
        client = groq.AsyncGroq(api_key=key)
        params = {**self._default_llm_params, **kwargs}
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            **params,
        )
        return resp.choices[0].message.content or ""

    async def _raw_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        groq = _import_groq()
        client = groq.AsyncGroq(api_key=key)
        params = {**self._default_llm_params, **kwargs}
        async with client.chat.completions.stream(
            model=model,
            messages=messages,
            **params,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

    # ------------------------------------------------------------------
    # Audio — Transcription (Whisper)
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: AudioInput,
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 0.0,
        timestamp_granularities: Optional[List[str]] = None,
    ) -> str:
        """
        Transcribe audio to text using a Whisper model.

        Args:
            audio:           Path (str/Path), raw bytes, or file-like object.
            model:           Whisper model override.
            language:        ISO-639-1 code (e.g. "en", "de"). Auto-detect if None.
            prompt:          Optional context hint for the model.
            response_format: "text" | "json" | "verbose_json" | "vtt" | "srt"
            temperature:     Sampling temperature (0.0 = deterministic).
            timestamp_granularities: ["word"] | ["segment"] (verbose_json only).

        Returns:
            Transcribed text (or raw JSON/SRT/VTT string per response_format).
        """
        return asyncio.run(
            self.atranscribe(
                audio,
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
            )
        )

    async def atranscribe(
        self,
        audio: AudioInput,
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 0.0,
        timestamp_granularities: Optional[List[str]] = None,
    ) -> str:
        groq = _import_groq()
        key = await self._rotator.acquire()
        try:
            client = groq.AsyncGroq(api_key=key)
            file_tuple = await asyncio.to_thread(_prepare_audio, audio)

            kwargs: Dict[str, Any] = dict(
                model=model or self._whisper_model,
                file=file_tuple,
                response_format=response_format,
                temperature=temperature,
            )
            if language:
                kwargs["language"] = language
            if prompt:
                kwargs["prompt"] = prompt
            if timestamp_granularities:
                kwargs["timestamp_granularities"] = timestamp_granularities

            resp = await client.audio.transcriptions.create(**kwargs)
            self._rotator.release(key)
            return resp.text if hasattr(resp, "text") else str(resp)
        except Exception as exc:
            _handle_rate_limit(exc, self._rotator, key)
            self._rotator.release(key)
            raise

    # ------------------------------------------------------------------
    # Audio — Translation (Whisper → English)
    # ------------------------------------------------------------------

    def translate(
        self,
        audio: AudioInput,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 0.0,
    ) -> str:
        """
        Translate audio to English using a Whisper model.

        Args:
            audio:           Path (str/Path), raw bytes, or file-like object.
            model:           Whisper model override.
            prompt:          Optional context hint.
            response_format: "text" | "json" | "verbose_json" | "vtt" | "srt"
            temperature:     0.0 for deterministic output.

        Returns:
            English translation of the audio.
        """
        return asyncio.run(
            self.atranslate(
                audio,
                model=model,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
            )
        )

    async def atranslate(
        self,
        audio: AudioInput,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 0.0,
    ) -> str:
        groq = _import_groq()
        key = await self._rotator.acquire()
        try:
            client = groq.AsyncGroq(api_key=key)
            file_tuple = await asyncio.to_thread(_prepare_audio, audio)

            kwargs: Dict[str, Any] = dict(
                model=model or self._whisper_model,
                file=file_tuple,
                response_format=response_format,
                temperature=temperature,
            )
            if prompt:
                kwargs["prompt"] = prompt

            resp = await client.audio.translations.create(**kwargs)
            self._rotator.release(key)
            return resp.text if hasattr(resp, "text") else str(resp)
        except Exception as exc:
            _handle_rate_limit(exc, self._rotator, key)
            self._rotator.release(key)
            raise

    # ------------------------------------------------------------------
    # Whisper model selector
    # ------------------------------------------------------------------

    def set_whisper_model(self, model: str) -> None:
        """Change the default Whisper model used for audio tasks."""
        self._whisper_model = model
        logger.info("Whisper model set to: %s", model)

    def key_status(self) -> List[dict]:
        return self._rotator.status()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_groq() -> Any:
    try:
        import groq  # type: ignore

        return groq
    except ImportError:
        raise ImportError("groq SDK required: pip install groq")


def _prepare_audio(audio: AudioInput) -> tuple:
    """
    Normalise various audio input types into (filename, bytes) tuple for
    the Groq audio API.
    """
    if isinstance(audio, (str, pathlib.Path)):
        p = pathlib.Path(audio)
        data = p.read_bytes()
        return (p.name, data)
    if isinstance(audio, bytes):
        return ("audio.mp3", audio)
    # File-like object
    data = audio.read()
    name = getattr(audio, "name", "audio.mp3")
    return (pathlib.Path(name).name, data)
