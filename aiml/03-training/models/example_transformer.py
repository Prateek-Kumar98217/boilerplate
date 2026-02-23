"""
Example lightweight Transformer encoder for sequence classification.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from models.base import BaseModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(BaseModel):
    """
    Transformer encoder + [CLS] token → classification head.

    Args:
        vocab_size:    Vocabulary size.
        num_classes:   Number of output classes.
        d_model:       Embedding dimension.
        nhead:         Number of attention heads.
        num_layers:    Number of encoder layers.
        dim_feedforward: FFN hidden dimension.
        max_seq_len:   Maximum sequence length.
        dropout:       Dropout rate.

    Example:
        model = TransformerClassifier(vocab_size=30522, num_classes=5)
        ids    = torch.randint(0, 30522, (4, 128))
        logits = model(ids)   # (4, 5)
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        num_classes: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) — 1 for real, 0 for pad
    ) -> torch.Tensor:
        # Build key padding mask: True where we want to IGNORE (padding=0 positions)
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = input_ids == 0

        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        cls = x[:, 0]  # Use first token as CLS representation
        return self.head(cls)

    def enable_gradient_checkpointing(self) -> None:
        import torch.utils.checkpoint as ckpt

        for layer in self.encoder.layers:
            orig = layer.forward
            layer.forward = lambda *a, fn=orig, **kw: ckpt.checkpoint(
                fn, *a, use_reentrant=False, **kw
            )
