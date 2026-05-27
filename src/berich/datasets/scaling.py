"""Minimal standard scaler fitted on training data only.

A deliberately tiny implementation (rather than sklearn) so the fit-on-train rule
is explicit and auditable: ``fit`` computes mean/std on the train fold, ``transform``
applies them to any fold. Reusing train statistics on the test fold is what prevents
lookahead leakage through normalization.
"""

from __future__ import annotations

import numpy as np


class StandardScaler:
    """Standardize columns to zero mean / unit variance using train-fold stats."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> StandardScaler:
        self.mean_ = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std[std == 0.0] = 1.0  # avoid divide-by-zero on constant columns
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            msg = "StandardScaler must be fit before transform"
            raise RuntimeError(msg)
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
