"""Common model interface.

Every model — the LightGBM baseline now, LSTM/TFT later — implements this tiny
protocol so the trainer, backtester, and signal service stay model-agnostic. The
target is binary (1 = winning swing long); ``predict_proba`` returns P(class == 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@runtime_checkable
class Model(Protocol):
    """Minimal fit/predict contract shared by all BeRich models."""

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
    ) -> Model:
        """Train on features ``x`` and binary labels ``y``."""
        ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return P(label == 1) as a 1-D array aligned to ``x``."""
        ...
