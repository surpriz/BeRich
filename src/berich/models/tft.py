"""Temporal Fusion Transformer-style model behind the :class:`Model` protocol.

A pure-``torch`` TFT in spirit (no ``pytorch-forecasting`` dependency, per CLAUDE.md rule
#4): per-step Gated Residual Network embedding, an LSTM encoder, interpretable multi-head
self-attention over the encoded sequence, and a gated skip connection into the output head.
It does not reproduce TFT's static-covariate / variable-selection machinery (we have no
static inputs here) but keeps its two signature ideas — gating and attention.

All sequence plumbing (per-ticker streams, train-fold scaling, ``_tail`` bootstrap, early
stopping, classifier-vs-regressor mode) is inherited from :class:`SequenceClassifier`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from berich.models.sequence_base import SequenceClassifier, SequenceConfig


@dataclass
class TFTConfig(SequenceConfig):
    """Hyperparameters for :class:`TFTModel`."""

    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 1  # LSTM encoder layers
    dropout: float = 0.2


class _GRN(nn.Module):
    """Gated Residual Network — the TFT building block (ELU + GLU gate + skip + norm)."""

    def __init__(self, input_size: int, hidden: int, *, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden, hidden * 2)
        self.skip = nn.Linear(input_size, hidden) if input_size != hidden else None
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.fc2(nn.functional.elu(self.fc1(x))))
        h = nn.functional.glu(self.gate(h), dim=-1)  # gated linear unit
        residual = x if self.skip is None else self.skip(x)
        return self.norm(residual + h)


class _TFTNet(nn.Module):
    """GRN embedding -> LSTM encoder -> self-attention -> gated head."""

    def __init__(
        self,
        *,
        n_features: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_grn = _GRN(n_features, d_model, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_grn = _GRN(d_model, d_model, dropout=dropout)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.input_grn(x)  # (B, L, d_model)
        encoded, _ = self.lstm(embedded)
        attended, _ = self.attn(encoded, encoded, encoded, need_weights=False)
        fused = self.attn_grn(encoded + attended)
        last = fused[:, -1, :]
        return self.head(self.head_dropout(last)).squeeze(-1)


class TFTModel(SequenceClassifier):
    """TFT-style classifier exposing the :class:`Model` protocol."""

    config_cls = TFTConfig
    cfg: TFTConfig

    def _build_net(self, n_features: int) -> nn.Module:
        return _TFTNet(
            n_features=n_features,
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        )


__all__ = ["TFTConfig", "TFTModel"]
