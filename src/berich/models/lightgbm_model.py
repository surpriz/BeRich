"""LightGBM baseline classifier.

Gradient-boosted trees on the tabular feature matrix. This is the bar every deep
model must clear: a model is only promoted if it beats this baseline *and* buy &
hold out-of-sample (the design's guard rule). Hyperparameters are sensible defaults;
Phase 3 tunes them with Optuna.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from lightgbm import LGBMClassifier

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_samples": 50,
    "n_jobs": -1,
    "verbosity": -1,
}


class LGBMModel:
    """Probabilistic binary classifier wrapping :class:`lightgbm.LGBMClassifier`."""

    def __init__(self, **params: Any) -> None:
        self.params = {**DEFAULT_PARAMS, **params}
        self._clf = LGBMClassifier(**self.params)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        tickers: pd.Series | None = None,  # noqa: ARG002 — tabular model ignores ticker grouping
    ) -> LGBMModel:
        weight = None if sample_weight is None else sample_weight.to_numpy()
        self._clf.fit(x, y, sample_weight=weight)
        return self

    def predict_proba(
        self,
        x: pd.DataFrame,
        *,
        tickers: pd.Series | None = None,  # noqa: ARG002 — tabular model ignores ticker grouping
    ) -> np.ndarray:
        proba = self._clf.predict_proba(x)
        # Column order follows classes_; return the probability of the positive class.
        pos = list(self._clf.classes_).index(1)
        return np.asarray(proba[:, pos], dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self._clf.feature_importances_)
