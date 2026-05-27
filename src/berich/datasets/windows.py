"""Sliding-window sequence builder for recurrent / attention models.

Tabular models (LightGBM) consume one feature row per sample. Sequence models
(LSTM, TFT) need a ``(lookback, n_features)`` window ending at the prediction bar.
`make_sequences` turns a 2-D feature matrix into a 3-D tensor without crossing the
sample boundary, keeping each window strictly causal (ends at the labeled bar).
"""

from __future__ import annotations

import numpy as np


def make_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build overlapping windows of length ``lookback``.

    Args:
        features: 2-D array ``(n_samples, n_features)``, time-ordered.
        labels: 1-D array ``(n_samples,)`` aligned to ``features``.
        lookback: number of bars per window.

    Returns:
        ``(X, y)`` where ``X`` is ``(n_windows, lookback, n_features)`` and ``y`` is
        ``(n_windows,)`` — the label of the bar at the end of each window.
    """
    n = len(features)
    if lookback <= 0:
        msg = "lookback must be positive"
        raise ValueError(msg)
    if n < lookback:
        n_features = features.shape[1] if features.ndim == 2 else 0  # noqa: PLR2004
        return np.empty((0, lookback, n_features)), np.empty((0,))

    windows = [features[i - lookback : i] for i in range(lookback, n + 1)]
    x = np.stack(windows)
    y = labels[lookback - 1 :]
    return x, y
