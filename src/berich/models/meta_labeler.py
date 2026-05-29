"""Meta-labeling model (López de Prado): predict whether to ACT on a primary signal.

The primary model produces P(win) and a side (BUY when proba >= threshold). The meta
model is a precision filter: given the features *plus the primary proba*, it predicts the
probability that acting on the BUY would actually win. A low meta-proba vetoes the BUY
(downgrades it to NEUTRAL) so we trade fewer, higher-precision setups.

It implements the :class:`~berich.models.base.Model` protocol (``fit`` / ``predict_proba``)
so it stores in the registry and trains through ``oof_predict`` like any other model. The
extra ``primary_proba`` meta-feature MUST be an out-of-fold primary prediction at fit time
(see :func:`berich.training.meta.build_meta_dataset`) — that is the leak guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from berich.models.lightgbm_model import LGBMModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

PRIMARY_PROBA_COL = "primary_proba"


class MetaLabeler:
    """LightGBM precision filter over base features + the primary proba."""

    def __init__(self, base_features: list[str], **params: Any) -> None:
        self.base_features = list(base_features)
        self.meta_columns = [*self.base_features, PRIMARY_PROBA_COL]
        self._model = LGBMModel(**params)

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        tickers: pd.Series | None = None,  # noqa: ARG002 — tabular model ignores grouping
    ) -> MetaLabeler:
        self._model.fit(x[self.meta_columns], y, sample_weight=sample_weight)
        return self

    def predict_proba(
        self,
        x: pd.DataFrame,
        *,
        tickers: pd.Series | None = None,  # noqa: ARG002 — tabular model ignores grouping
    ) -> np.ndarray:
        return self._model.predict_proba(x[self.meta_columns])


__all__ = ["PRIMARY_PROBA_COL", "MetaLabeler"]
