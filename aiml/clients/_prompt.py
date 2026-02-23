"""
Customisable prompt template engine for inference clients.

Features:
  • Variable interpolation with {placeholder} syntax
  • Pre-built templates for common patterns (QA, chat, RAG, structured output,
    chain-of-thought, classification, summarisation, code generation)
  • Template composition: build complex prompts from reusable blocks
  • Runtime override of any variable at call time
  • Validation of required variables before rendering

Usage:
    # Simple template
    tmpl = PromptTemplate(
        system="You are a {role}. Answer {style}.",
        user="{question}",
    )
    msgs = tmpl.render(role="doctor", style="concisely", question="What is aspirin?")

    # Pre-built
    msgs = Templates.rag_qa.render(context="...", question="What is FAISS?")
"""

from __future__ import annotations

import re
import string
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core template class
# ---------------------------------------------------------------------------


@dataclass
class PromptTemplate:
    """
    A reusable prompt template with variable substitution.

    Args:
        system:   System message template (optional).
        user:     User message template (optional, used for simple completions).
        turns:    Ordered list of (role, template_str) for multi-turn templates.
        defaults: Default values for template variables.
        name:     Human-readable identifier.

    Variables in templates use Python's {variable_name} syntax.
    """

    system: Optional[str] = None
    user: Optional[str] = None
    turns: List[tuple[str, str]] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    name: str = "custom"

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, **kwargs: Any) -> List[Dict[str, str]]:
        """
        Render the template into a messages list.

        Args:
            **kwargs: Variable values, overriding any defaults.

        Returns:
            List of {"role": ..., "content": ...} dicts.

        Raises:
            KeyError: If a required variable is missing and no default exists.
        """
        ctx = {**self.defaults, **kwargs}
        messages = []

        if self.system:
            messages.append({"role": "system", "content": self._fill(self.system, ctx)})

        for role, tmpl in self.turns:
            messages.append({"role": role, "content": self._fill(tmpl, ctx)})

        if self.user:
            messages.append({"role": "user", "content": self._fill(self.user, ctx)})

        return messages

    def render_system(self, **kwargs: Any) -> str:
        """Render only the system string (useful for injecting into memory)."""
        ctx = {**self.defaults, **kwargs}
        return self._fill(self.system or "", ctx)

    def render_user(self, **kwargs: Any) -> str:
        ctx = {**self.defaults, **kwargs}
        return self._fill(self.user or "", ctx)

    def variables(self) -> List[str]:
        """Return all placeholder variable names across all template strings."""
        parts = []
        if self.system:
            parts.append(self.system)
        if self.user:
            parts.append(self.user)
        for _, t in self.turns:
            parts.append(t)
        found = set()
        for part in parts:
            found.update(re.findall(r"\{(\w+)\}", part))
        return sorted(found)

    def required_variables(self) -> List[str]:
        """Variables without defaults — MUST be supplied at render time."""
        return [v for v in self.variables() if v not in self.defaults]

    def with_defaults(self, **overrides: Any) -> "PromptTemplate":
        """Return a new template with updated default variables."""
        new = deepcopy(self)
        new.defaults.update(overrides)
        return new

    def extend_system(self, extra: str) -> "PromptTemplate":
        """Return a new template with appended text on the system prompt."""
        new = deepcopy(self)
        new.system = ((new.system or "") + "\n\n" + extra).strip()
        return new

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill(template: str, ctx: Dict[str, Any]) -> str:
        """
        Fill {placeholders} using Python str.format_map.
        Missing keys raise a helpful KeyError.
        """
        try:
            return template.format_map(ctx)
        except KeyError as exc:
            raise KeyError(
                f"PromptTemplate: missing variable {exc} in template. "
                f"Available: {list(ctx.keys())}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"PromptTemplate(name={self.name!r}, "
            f"vars={self.variables()}, "
            f"required={self.required_variables()})"
        )


# ---------------------------------------------------------------------------
# TemplateLibrary — built-in templates, accessible as Templates.name
# ---------------------------------------------------------------------------


class _TemplateLibrary:
    """Namespace of pre-built templates."""

    # ── Plain chat / completion ───────────────────────────────────────────

    chat = PromptTemplate(
        name="chat",
        system="You are a helpful, accurate, and concise AI assistant.",
        user="{message}",
    )

    completion = PromptTemplate(
        name="completion",
        user="{prompt}",
    )

    # ── Question Answering ────────────────────────────────────────────────

    qa = PromptTemplate(
        name="qa",
        system=(
            "You are a knowledgeable assistant. "
            "Answer questions accurately and concisely."
        ),
        user="{question}",
    )

    # ── RAG ───────────────────────────────────────────────────────────────

    rag_qa = PromptTemplate(
        name="rag_qa",
        system=(
            "You are a helpful assistant. "
            "Answer ONLY using the context provided below. "
            'If the context does not contain the answer, say "I don\'t know."'
        ),
        user="Context:\n{context}\n\nQuestion: {question}",
    )

    rag_citation = PromptTemplate(
        name="rag_citation",
        system=(
            "You are a research assistant. "
            "Answer using the context and cite sources with [Source N]."
        ),
        user="Context:\n{context}\n\nQuestion: {question}",
    )

    # ── Chain-of-Thought ──────────────────────────────────────────────────

    chain_of_thought = PromptTemplate(
        name="chain_of_thought",
        system=(
            "You are a careful reasoner. "
            "First think step by step inside <think>...</think> tags, "
            "then give your final answer."
        ),
        user="{question}",
    )

    # ── Classification ────────────────────────────────────────────────────

    classification = PromptTemplate(
        name="classification",
        system=(
            "You are a text classifier. "
            "Classify the input into exactly one of the following categories: {categories}. "
            "Reply with ONLY the category name, nothing else."
        ),
        user="{text}",
    )

    # ── Summarisation ─────────────────────────────────────────────────────

    summarise = PromptTemplate(
        name="summarise",
        system=(
            "You are a summarisation expert. "
            "Summarise the text {style}. "
            "Preserve all key facts and figures."
        ),
        user="Text to summarise:\n{text}",
        defaults={"style": "concisely in 3-5 sentences"},
    )

    bullet_summary = PromptTemplate(
        name="bullet_summary",
        system="You are a concise summariser. Reply with a bullet-point list.",
        user="Summarise the following:\n{text}",
    )

    # ── Code generation ───────────────────────────────────────────────────

    code_gen = PromptTemplate(
        name="code_gen",
        system=(
            "You are an expert {language} programmer. "
            "Write clean, well-commented code. "
            "Reply with only the code inside a ```{language} ... ``` block."
        ),
        user="{task}",
        defaults={"language": "Python"},
    )

    code_review = PromptTemplate(
        name="code_review",
        system=(
            "You are a senior {language} engineer. "
            "Review the code below for bugs, style issues, and improvements."
        ),
        user="```{language}\n{code}\n```",
        defaults={"language": "Python"},
    )

    # ── Structured JSON output ────────────────────────────────────────────

    json_extract = PromptTemplate(
        name="json_extract",
        system=(
            "You are a precise information extractor. "
            "Extract the requested data from the text and return ONLY valid JSON "
            "matching this schema:\n{schema}\n"
            "Do not include any explanation."
        ),
        user="Text:\n{text}\n\nExtraction request: {request}",
    )

    # ── Translation ───────────────────────────────────────────────────────

    translate = PromptTemplate(
        name="translate",
        system=(
            "You are a professional translator. "
            "Translate the text to {target_language}. "
            "Preserve tone and formatting."
        ),
        user="{text}",
    )

    # ── Persona / role-play ───────────────────────────────────────────────

    persona = PromptTemplate(
        name="persona",
        system="{persona_description}",
        user="{message}",
    )

    # ── Audio transcription post-processing ───────────────────────────────

    transcript_clean = PromptTemplate(
        name="transcript_clean",
        system=(
            "You are a transcription editor. "
            "Clean up the raw transcript below: fix grammar, remove filler words, "
            "and add punctuation. Preserve the original meaning."
        ),
        user="{transcript}",
    )


Templates = _TemplateLibrary()
