"""PatchTST-style transformer classifier behind the :class:`Model` protocol.

A patch-embedding transformer over the lookback window: the sequence is split into
overlapping patches, each patch (flattened across features) is linearly embedded, a
positional embedding is added, a standard transformer encoder mixes the patches, and a
mean-pooled representation feeds a linear head to a scalar logit. Pure ``torch`` — no
``pytorch-forecasting`` dependency, so VRAM and shapes are fully controllable.

All sequence plumbing (per-ticker streams, train-fold-only scaling, ``_tail`` bootstrap,
early stopping, batched inference) is inherited from :class:`SequenceClassifier`; this
module only supplies the network and its hyperparameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from berich.models.sequence_base import SequenceClassifier, SequenceConfig


@dataclass
class PatchTSTConfig(SequenceConfig):
    """Hyperparameters for :class:`PatchTSTModel`."""

    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 8
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.2


class _PatchTSTNet(nn.Module):
    """Patch embedding + transformer encoder + mean-pool + linear head."""

    def __init__(
        self,
        *,
        n_features: int,
        lookback: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if patch_len > lookback:
            msg = f"patch_len {patch_len} exceeds lookback {lookback}"
            raise ValueError(msg)
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (lookback - patch_len) // stride + 1

        self.embed = nn.Linear(patch_len * n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features) -> patches: (batch, n_patches, patch_len*n_features)
        patches = [
            x[:, i * self.stride : i * self.stride + self.patch_len, :].reshape(x.shape[0], -1)
            for i in range(self.n_patches)
        ]
        tokens = torch.stack(patches, dim=1)
        tokens = self.embed(tokens) * math.sqrt(self.embed.out_features) + self.pos_embed
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(self.head_dropout(pooled)).squeeze(-1)


class PatchTSTModel(SequenceClassifier):
    """PatchTST classifier exposing the :class:`Model` protocol."""

    config_cls = PatchTSTConfig
    cfg: PatchTSTConfig

    def _build_net(self, n_features: int) -> nn.Module:
        return _PatchTSTNet(
            n_features=n_features,
            lookback=self.cfg.lookback,
            patch_len=self.cfg.patch_len,
            stride=self.cfg.stride,
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            num_layers=self.cfg.num_layers,
            dim_feedforward=self.cfg.dim_feedforward,
            dropout=self.cfg.dropout,
        )


__all__ = ["PatchTSTConfig", "PatchTSTModel"]
