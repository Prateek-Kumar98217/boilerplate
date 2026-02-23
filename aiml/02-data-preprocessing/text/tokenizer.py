"""
HuggingFace tokenizer wrapper with padding, truncation, and batch encoding.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, BatchEncoding


class HFTokenizer:
    """
    Wraps AutoTokenizer with sensible defaults for training/inference.

    Example:
        tok = HFTokenizer("bert-base-uncased", max_length=128)
        enc = tok(["Hello world", "Boilerplate rocks"])
        # enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 512,
        padding: str = "max_length",  # max_length | longest | do_not_pad
        truncation: bool = True,
        return_tensors: str = "pt",  # pt | tf | np | None
        add_special_tokens: bool = True,
        token: Optional[str] = None,
    ) -> None:
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.add_special_tokens = add_special_tokens

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._tokenizer.pad_token_id

    def encode(self, texts: Union[str, List[str]]) -> BatchEncoding:
        """Tokenize one or multiple texts."""
        return self._tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            add_special_tokens=self.add_special_tokens,
        )

    def decode(
        self, token_ids: torch.Tensor, skip_special_tokens: bool = True
    ) -> List[str]:
        return self._tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def __call__(self, texts: Union[str, List[str]]) -> BatchEncoding:
        return self.encode(texts)

    # ── Sliding window for long documents ────────────────────────────
    def encode_long_document(
        self, text: str, stride: int = 128
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Chunk a long document into overlapping windows using stride.
        Returns a list of BatchEncoding chunks.
        """
        tokens = self._tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )
        chunks = []
        for i in range(tokens["input_ids"].shape[0]):
            chunks.append({k: v[i] for k, v in tokens.items()})
        return chunks
