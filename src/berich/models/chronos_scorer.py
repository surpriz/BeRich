"""Optional zero-shot time-series foundation model (Amazon Chronos-Bolt).

Chronos-Bolt is a pretrained forecaster: no training, just condition on a price history and
sample a forward distribution. We use it zero-shot to produce forward-return **quantiles**
(which feed the adaptive SL/TP path) and a directional score. Chronos-Bolt-small (~48M
params) fits comfortably on a 16-24 GB GPU.

This is the riskiest, most optional zoo member, so it is fully guarded: the heavy
``chronos`` package is imported lazily and a clear, actionable error is raised if it is not
installed (``uv add chronos-forecasting``). Nothing else in the codebase imports this module
eagerly, so a missing dependency never breaks the rest of the pipeline.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "amazon/chronos-bolt-small"


class ChronosForecaster:
    """Zero-shot forward-return quantile forecaster backed by Chronos-Bolt."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, device: str | None = None) -> None:
        # importlib (rather than bare imports) keeps these optional deps out of the
        # linter's import-ordering / top-level-import rules while staying fully guarded.
        try:
            torch = importlib.import_module("torch")
            chronos = importlib.import_module("chronos")
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            msg = (
                "ChronosForecaster requires the optional 'chronos-forecasting' package. "
                "Install it with `uv add chronos-forecasting` (or pip install chronos-forecasting)."
            )
            raise ImportError(msg) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self.pipeline = chronos.BaseChronosPipeline.from_pretrained(
            model_name, device_map=self.device
        )

    def predict_quantiles(
        self,
        close: pd.Series,
        *,
        horizon_days: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> np.ndarray:
        """Cumulative forward-return quantiles over ``horizon_days`` (zero-shot).

        Forecasts the price path, takes the quantiles of ``close[t+H]/close[t] - 1``.
        Returns a 1-D array aligned to ``quantiles``.
        """
        prices = close.astype(float).to_numpy()
        last = float(prices[-1])
        context = self._torch.tensor(prices, dtype=self._torch.float32)
        q_levels = list(quantiles)
        # Chronos-Bolt returns (quantiles[batch, horizon, n_quantiles], mean[batch, horizon]).
        q_pred, _mean = self.pipeline.predict_quantiles(
            context=context, prediction_length=horizon_days, quantile_levels=q_levels
        )
        terminal = q_pred[0, -1, :].cpu().numpy()  # quantiles of the price at t+H
        return np.asarray(terminal / last - 1.0, dtype=float)

    def direction_score(self, close: pd.Series, *, horizon_days: int) -> float:
        """A simple directional score in [0, 1]: median forecast return mapped through 0.5+.

        Convenient as a zero-shot ranking signal; not calibrated as a probability.
        """
        q = self.predict_quantiles(close, horizon_days=horizon_days, quantiles=(0.5,))
        median_ret = float(q[0])
        return float(1.0 / (1.0 + np.exp(-50.0 * median_ret)))  # squashes ~±2% to a [0,1] score


__all__ = ["DEFAULT_MODEL", "ChronosForecaster"]
