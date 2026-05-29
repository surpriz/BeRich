"""LightGBM regression ranker for the cross-sectional long/short track.

The cross-sectional target is a continuous (beta-residualized, z-scored) forward
return, so we wrap :class:`lightgbm.LGBMRegressor` rather than the classifier. To stay
behind the existing :class:`~berich.models.base.Model` protocol — which every reusable
seam (``oof_predict``, the registry, drift monitoring) speaks — ``predict_proba`` returns
the raw regression **score**. Downstream code only ever *ranks* this score within a date,
so any monotone transform works and absolute calibration is irrelevant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from lightgbm import LGBMRegressor

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


class LGBMRanker:
    """Regression ranker exposing the :class:`Model` protocol (``predict_proba`` = score)."""

    def __init__(self, **params: Any) -> None:
        self.params = {**DEFAULT_PARAMS, **params}
        self._reg = LGBMRegressor(**self.params)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        tickers: pd.Series | None = None,  # noqa: ARG002 — tabular model ignores ticker grouping
    ) -> LGBMRanker:
        weight = None if sample_weight is None else sample_weight.to_numpy()
        self._reg.fit(x, y, sample_weight=weight)
        return self

    def predict_proba(
        self,
        x: pd.DataFrame,
        *,
        tickers: pd.Series | None = None,  # noqa: ARG002 — tabular model ignores ticker grouping
    ) -> np.ndarray:
        """Return the raw regression score (a monotone ranking signal, not a probability)."""
        return np.asarray(self._reg.predict(x), dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self._reg.feature_importances_)


__all__ = ["LGBMRanker"]
