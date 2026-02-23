"""
Text chunkers — split documents into retrieval-ready chunks.

Strategies:
- RecursiveCharacterTextSplitter (default, best general purpose)
- SentenceSplitter (semantic boundaries)
- TokenSplitter (exact token count via tiktoken)
- FixedSizeSplitter (simple fixed char count)
"""

from __future__ import annotations

import re
from typing import List, Optional

from ingestion.document_loader import Document


# ── Base ──────────────────────────────────────────────────────────────


class BaseChunker:
    def split_documents(self, docs: List[Document]) -> List[Document]:
        result = []
        for doc in docs:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                result.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "chunk_count": len(chunks),
                        },
                    )
                )
        return result

    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError


# ── Recursive splitter ────────────────────────────────────────────────


class RecursiveCharacterSplitter(BaseChunker):
    """
    Splits text recursively by paragraph → sentence → word boundaries.
    Best general-purpose chunker for most document types.

    Example:
        splitter = RecursiveCharacterSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split_documents(docs)
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self._SEPARATORS)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return self._merge([text])

        sep = separators[0]
        splits = text.split(sep)
        good, too_long = [], []
        for s in splits:
            if len(s) <= self.chunk_size:
                good.append(s)
            else:
                too_long.extend(self._split(s, separators[1:]))

        # Merge alongside overlap
        return self._merge(good + too_long, sep)

    def _merge(self, splits: List[str], sep: str = "") -> List[str]:
        chunks, current, current_len = [], [], 0
        for split in splits:
            split = split.strip()
            if not split:
                continue
            slen = len(split)
            if current_len + slen + len(sep) > self.chunk_size and current:
                chunks.append(sep.join(current))
                # Overlap: keep last N characters worth of splits
                overlap_len = 0
                keep = []
                for s in reversed(current):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        keep.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = keep
                current_len = overlap_len
            current.append(split)
            current_len += slen
        if current:
            chunks.append(sep.join(current))
        return [c for c in chunks if c.strip()]


# ── Sentence splitter ─────────────────────────────────────────────────


class SentenceSplitter(BaseChunker):
    """
    Split by sentences using regex, then group into chunk_size windows.

    Example:
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=2)
        # chunk_overlap here = number of sentences to overlap
    """

    _SENT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 2) -> None:
        self.chunk_size = chunk_size
        self.sentence_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        sentences = [s.strip() for s in self._SENT_RE.split(text) if s.strip()]
        chunks, current, current_len = [], [], 0
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            if current_len + len(sent) > self.chunk_size and current:
                chunks.append(" ".join(current))
                # Overlap
                current = (
                    current[-self.sentence_overlap :] if self.sentence_overlap else []
                )
                current_len = sum(len(s) for s in current)
            current.append(sent)
            current_len += len(sent)
            i += 1
        if current:
            chunks.append(" ".join(current))
        return chunks


# ── Token splitter ────────────────────────────────────────────────────


class TokenSplitter(BaseChunker):
    """
    Split by exact token count using tiktoken.
    Install: pip install tiktoken

    Example:
        splitter = TokenSplitter(model="gpt-4", chunk_size=512, chunk_overlap=50)
    """

    def __init__(
        self, model: str = "gpt-4o", chunk_size: int = 512, chunk_overlap: int = 50
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            import tiktoken

            self._enc = tiktoken.encoding_for_model(model)
        except ImportError:
            raise ImportError("tiktoken required: pip install tiktoken")

    def split_text(self, text: str) -> List[str]:
        tokens = self._enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(self._enc.decode(tokens[start:end]))
            start += self.chunk_size - self.chunk_overlap
        return chunks


# ── Factory ───────────────────────────────────────────────────────────


def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> BaseChunker:
    """
    Get a chunker by strategy name.

    Args:
        strategy: recursive | sentence | token | fixed
    """
    if strategy == "recursive":
        return RecursiveCharacterSplitter(chunk_size, chunk_overlap)
    elif strategy == "sentence":
        return SentenceSplitter(chunk_size, chunk_overlap=max(1, chunk_overlap // 80))
    elif strategy == "token":
        return TokenSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
