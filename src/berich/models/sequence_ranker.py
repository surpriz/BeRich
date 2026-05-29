"""GPU sequence *rankers* for the market-neutral long/short path.

Same sequence plumbing as the classifiers, but trained with MSE on the continuous
cross-sectional target (the within-date z-scored residual) instead of BCE on a binary
label. ``predict_proba`` returns the raw predicted score — a monotone ranking signal that
:func:`berich.training.cross_sectional.oof_predict_cross_sectional` and the long/short
backtester consume (only the within-date rank matters, so absolute scale is irrelevant).
"""

from __future__ import annotations

import torch
from torch import nn

from berich.models.lstm import LSTMConfig, _LSTMNet
from berich.models.patchtst import PatchTSTConfig, _PatchTSTNet
from berich.models.sequence_base import SequenceClassifier


class SequenceRegressor(SequenceClassifier):
    """Sequence model trained with MSE; outputs a raw score (no sigmoid)."""

    is_classifier = False
    neutral_fallback = 0.0  # neutral cross-sectional score for tickers without history

    def _criterion(self) -> nn.Module:
        return nn.MSELoss()

    def _output_transform(self, logits: torch.Tensor) -> torch.Tensor:
        return logits  # raw score (no sigmoid) — ranking signal


class LSTMRanker(SequenceRegressor):
    """LSTM ranker for cross-sectional forward-return prediction."""

    config_cls = LSTMConfig
    cfg: LSTMConfig

    def _build_net(self, n_features: int) -> nn.Module:
        return _LSTMNet(
            n_features=n_features,
            hidden=self.cfg.hidden,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        )


class PatchTSTRanker(SequenceRegressor):
    """PatchTST ranker for cross-sectional forward-return prediction."""

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


__all__ = ["LSTMRanker", "PatchTSTRanker", "SequenceRegressor"]
