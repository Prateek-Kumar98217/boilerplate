"""
Google Gemini inference client.

Install: pip install google-generativeai

Supported input types:
  • Text chat / completion            — GeminiClient.chat() / complete()
  • Vision (image + text)             — GeminiClient.vision()
  • Audio analysis / transcription    — GeminiClient.audio()
  • Multimodal (mix of parts)         — GeminiClient.multimodal()
  • Text embeddings                   — GeminiClient.embed() / embed_batch()

Key rotation:
  Pass multiple API keys as GEMINI_API_KEYS=key1,key2,key3 in .env
  or supply a list directly to the constructor.

Memory:
  Automatic short-term + long-term summarisation is enabled by default.
  Access via client.memory.  Disable with enable_memory=False.

Prompt templates:
  client.set_template(Templates.rag_qa)
  client.set_system("You are a medical expert.")
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import pathlib
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Union

from _base import BaseClient, _handle_rate_limit
from _key_rotator import KeyRotator
from _prompt import PromptTemplate, Templates

logger = logging.getLogger(__name__)

# Type alias for image inputs
ImageInput = Union[str, pathlib.Path, bytes, "PIL.Image.Image"]  # type: ignore[name-defined]


class GeminiClient(BaseClient):
    """
    Full-featured Gemini client with key rotation, memory, and prompt templates.

    Args:
        api_keys:       One or more Gemini API keys (rotated automatically).
        model:          Default text model (default: gemini-2.0-flash).
        vision_model:   Model used for vision tasks.
        embedding_model: Model used for embeddings.
        rpm_limit:      Requests-per-minute per key for rate limiting.
        rpd_limit:      Requests-per-day per key.
        system_prompt:  Global system instruction.
        template:       Default prompt template.
        memory_max_turns: Max turns kept verbatim in short-term memory.
        memory_max_chars: Char limit for short-term window.
        enable_memory:  Toggle memory on/off.
        generation_config: Default params (temperature, top_p, max_tokens, etc.).

    Example:
        client = GeminiClient.from_env()
        reply = client.chat("Explain transformers in 3 sentences.")

        # Vision
        reply = client.vision("Describe this image.", image="path/to/img.png")

        # Embedding
        vec = client.embed("Hello, world!")
    """

    def __init__(
        self,
        api_keys: List[str],
        model: str = "gemini-2.0-flash",
        vision_model: str = "gemini-2.0-flash",
        embedding_model: str = "text-embedding-004",
        rpm_limit: int = 15,
        rpd_limit: int = 1500,
        system_prompt: Optional[str] = None,
        template: Optional[PromptTemplate] = None,
        memory_max_turns: int = 20,
        memory_max_chars: int = 8000,
        enable_memory: bool = True,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        rotator = KeyRotator(
            keys=api_keys,
            rpm_limit=rpm_limit,
            rpd_limit=rpd_limit,
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
        self._vision_model = vision_model
        self._embedding_model = embedding_model
        self._gen_config = generation_config or {}
        self._sdk_cache: Dict[str, Any] = {}  # key → genai.GenerativeModel

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, **kwargs: Any) -> "GeminiClient":
        """Construct from environment / .env file."""
        from config import get_settings

        cfg = get_settings().gemini
        keys = cfg.api_keys or [os.environ.get("GEMINI_API_KEY", "")]
        return cls(
            api_keys=[k for k in keys if k],
            model=cfg.default_model,
            vision_model=cfg.vision_model,
            embedding_model=cfg.embedding_model,
            rpm_limit=cfg.rpm_limit,
            rpd_limit=cfg.rpd_limit,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Text — overrides BaseClient abstract method
    # ------------------------------------------------------------------

    @property
    def default_model(self) -> str:
        return "gemini-2.0-flash"

    async def _raw_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str,
        **kwargs: Any,
    ) -> str:
        genai = _import_genai()
        genai.configure(api_key=key)
        system_instruction, history, last_user = _split_messages(messages)
        sdk_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction or None,
            generation_config={**self._gen_config, **kwargs},
        )
        chat = sdk_model.start_chat(history=history)
        resp = await asyncio.to_thread(chat.send_message, last_user)
        return resp.text

    async def _raw_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        key: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        genai = _import_genai()
        genai.configure(api_key=key)
        system_instruction, history, last_user = _split_messages(messages)
        sdk_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction or None,
            generation_config={**self._gen_config, **kwargs},
        )
        chat = sdk_model.start_chat(history=history)

        def _stream():
            return chat.send_message(last_user, stream=True)

        response = await asyncio.to_thread(_stream)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    def vision(
        self,
        prompt: str,
        image: ImageInput,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Analyse an image with an optional text prompt.

        Args:
            prompt: Text instruction / question about the image.
            image:  File path (str/Path), raw bytes, or PIL Image.
            model:  Override the vision model.
        """
        return asyncio.run(self.avision(prompt, image, model=model, **kwargs))

    async def avision(
        self,
        prompt: str,
        image: ImageInput,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        genai = _import_genai()
        key = await self._rotator.acquire()
        genai.configure(api_key=key)
        used_model = model or self._vision_model
        try:
            img_part = _prepare_image_part(image, genai)
            sdk_model = genai.GenerativeModel(
                model_name=used_model,
                generation_config={**self._gen_config, **kwargs},
            )
            resp = await asyncio.to_thread(
                sdk_model.generate_content, [img_part, prompt]
            )
            self._rotator.release(key)
            return resp.text
        except Exception as exc:
            _handle_rate_limit(exc, self._rotator, key)
            self._rotator.release(key)
            raise

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def audio(
        self,
        prompt: str,
        audio_path: Union[str, pathlib.Path],
        mime_type: str = "audio/mp3",
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Analyse / transcribe an audio file with Gemini.

        Args:
            prompt:     Instruction (e.g. "Transcribe this audio.").
            audio_path: Path to the audio file.
            mime_type:  MIME type (audio/mp3, audio/wav, audio/ogg, etc.).
            model:      Model override.
        """
        return asyncio.run(
            self.aaudio(prompt, audio_path, mime_type, model=model, **kwargs)
        )

    async def aaudio(
        self,
        prompt: str,
        audio_path: Union[str, pathlib.Path],
        mime_type: str = "audio/mp3",
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        genai = _import_genai()
        key = await self._rotator.acquire()
        genai.configure(api_key=key)
        used_model = model or self._model
        try:
            audio_bytes = pathlib.Path(audio_path).read_bytes()
            audio_part = {"mime_type": mime_type, "data": audio_bytes}
            sdk_model = genai.GenerativeModel(
                model_name=used_model,
                generation_config={**self._gen_config, **kwargs},
            )
            resp = await asyncio.to_thread(
                sdk_model.generate_content, [audio_part, prompt]
            )
            self._rotator.release(key)
            return resp.text
        except Exception as exc:
            _handle_rate_limit(exc, self._rotator, key)
            self._rotator.release(key)
            raise

    # ------------------------------------------------------------------
    # Multimodal (arbitrary mix of text + images + audio parts)
    # ------------------------------------------------------------------

    def multimodal(
        self,
        parts: List[Any],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Send a list of mixed content parts to Gemini.

        Each element in ``parts`` can be:
          • A string  → treated as a text part
          • bytes     → treated as an inline blob (must set mime_type separately)
          • A dict    → passed as-is {"mime_type": ..., "data": ...}
          • PIL Image → converted automatically

        Example:
            client.multimodal([
                "Compare these two images:",
                PIL.Image.open("img1.jpg"),
                PIL.Image.open("img2.jpg"),
            ])
        """
        return asyncio.run(self.amultimodal(parts, model=model, **kwargs))

    async def amultimodal(
        self,
        parts: List[Any],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        genai = _import_genai()
        key = await self._rotator.acquire()
        genai.configure(api_key=key)
        used_model = model or self._vision_model
        try:
            prepared = [
                _prepare_image_part(p, genai) if _is_image_like(p) else p for p in parts
            ]
            sdk_model = genai.GenerativeModel(
                model_name=used_model,
                generation_config={**self._gen_config, **kwargs},
            )
            resp = await asyncio.to_thread(sdk_model.generate_content, prepared)
            self._rotator.release(key)
            return resp.text
        except Exception as exc:
            _handle_rate_limit(exc, self._rotator, key)
            self._rotator.release(key)
            raise

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Return a dense embedding vector for a single text string."""
        return asyncio.run(self.aembed(text, model=model))

    async def aembed(self, text: str, model: Optional[str] = None) -> List[float]:
        genai = _import_genai()
        key = await self._rotator.acquire()
        genai.configure(api_key=key)
        used_model = model or self._embedding_model
        try:
            result = await asyncio.to_thread(
                genai.embed_content, model=used_model, content=text
            )
            self._rotator.release(key)
            return result["embedding"]
        except Exception as exc:
            _handle_rate_limit(exc, self._rotator, key)
            self._rotator.release(key)
            raise

    def embed_batch(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        """Embed multiple texts, returning a list of vectors."""
        return asyncio.run(self.aembed_batch(texts, model=model))

    async def aembed_batch(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        tasks = [self.aembed(t, model=model) for t in texts]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Rotator / key status
    # ------------------------------------------------------------------

    def key_status(self) -> List[dict]:
        return self._rotator.status()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_genai() -> Any:
    try:
        import google.generativeai as genai

        return genai
    except ImportError:
        raise ImportError(
            "google-generativeai required: pip install google-generativeai"
        )


def _split_messages(
    messages: List[Dict[str, str]],
) -> tuple[str, list, str]:
    """
    Split an OpenAI-style messages list into:
      system_instruction, history (Gemini format), last_user_message
    """
    system_parts = []
    history = []
    user_messages = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_messages.append(content)
        elif role == "assistant":
            # Pair with the last user message for Gemini history format
            if user_messages:
                history.append({"role": "user", "parts": [user_messages.pop(0)]})
            history.append({"role": "model", "parts": [content]})

    system_instruction = "\n\n".join(system_parts).strip()
    last_user = user_messages[-1] if user_messages else ""
    return system_instruction, history, last_user


def _prepare_image_part(image: Any, genai: Any) -> Any:
    """Convert various image inputs to a Gemini-compatible part."""
    try:
        from PIL import Image as PILImage  # type: ignore

        pil_available = True
    except ImportError:
        pil_available = False

    if pil_available and isinstance(image, __import__("PIL").Image.Image):  # type: ignore
        return image  # Gemini SDK accepts PIL images directly

    if isinstance(image, (str, pathlib.Path)):
        data = pathlib.Path(image).read_bytes()
        suffix = pathlib.Path(image).suffix.lower().lstrip(".")
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime = mime_map.get(suffix, "image/jpeg")
        return {"mime_type": mime, "data": data}

    if isinstance(image, bytes):
        return {"mime_type": "image/jpeg", "data": image}

    return image  # pass through if already a dict/part


def _is_image_like(part: Any) -> bool:
    try:
        from PIL import Image as PILImage  # type: ignore

        if isinstance(part, PILImage.Image):  # type: ignore
            return True
    except ImportError:
        pass
    return isinstance(part, (pathlib.Path, bytes)) or (
        isinstance(part, str)
        and any(
            part.lower().endswith(ext)
            for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp")
        )
    )
