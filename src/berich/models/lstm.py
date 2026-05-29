"""LSTM classifier behind the :class:`~berich.models.base.Model` protocol.

A two-layer LSTM with a linear head over the canonical feature panel. All the
sequence plumbing — per-ticker streams, train-fold-only scaling, the prediction
``_tail`` bootstrap, early stopping — lives in :class:`SequenceClassifier`; this
module only supplies the network and its hyperparameters.

Important: the input ``x`` must be already filtered to ``FEATURE_COLUMNS`` order
(this is what :func:`berich.datasets.build_dataset` produces). ``tickers`` is a
Series aligned to ``x.index`` whose values identify which symbol each row belongs
to. When ``tickers`` is ``None`` the whole frame is treated as a single stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from berich.models.sequence_base import SequenceClassifier, SequenceConfig


@dataclass
class LSTMConfig(SequenceConfig):
    """Hyperparameters for :class:`LSTMModel` — small enough to be exhaustively logged."""

    hidden: int = 128
    num_layers: int = 2
    dropout: float = 0.2


class _LSTMNet(nn.Module):
    """Stacked LSTM with dropout between layers + linear head to a scalar logit."""

    def __init__(self, *, n_features: int, hidden: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch only applies inter-layer dropout when num_layers > 1.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.head_dropout(last)).squeeze(-1)


class LSTMModel(SequenceClassifier):
    """LSTM classifier exposing the :class:`Model` protocol."""

    config_cls = LSTMConfig
    cfg: LSTMConfig

    def _build_net(self, n_features: int) -> nn.Module:
        return _LSTMNet(
            n_features=n_features,
            hidden=self.cfg.hidden,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        )


__all__ = ["LSTMConfig", "LSTMModel"]
