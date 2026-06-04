"""Leak-free stacking ensemble behind the :class:`~berich.models.base.Model` protocol.

Combines several base models into one. The meta-learner is trained on **out-of-fold** base
predictions (walk-forward, with the same embargo as the rest of the pipeline) so it never
sees a leaked in-sample prediction — the textbook stacking-with-time-series-CV recipe. Base
models are then refit on all data for serving.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from berich.datasets.splits import walk_forward_splits

if TYPE_CHECKING:
    import pandas as pd

    from berich.models.base import Model

ModelFactory = Callable[[], "Model"]
MetaFactory = Callable[[], object]
# A member of an AveragingEnsemble: how to build the base model + the feature columns it consumes.
MemberSpec = tuple[ModelFactory, list[str]]


def _default_meta() -> object:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    return LogisticRegression(max_iter=1000)


class StackingEnsemble:
    """Stack several base models with a leak-free out-of-fold meta-learner."""

    def __init__(
        self,
        base_factories: list[ModelFactory],
        *,
        meta_factory: MetaFactory = _default_meta,
        train_frac: float = 0.5,
        test_frac: float = 0.1,
        embargo: int = 10,
    ) -> None:
        if not base_factories:
            msg = "StackingEnsemble needs at least one base model factory"
            raise ValueError(msg)
        self.base_factories = base_factories
        self.meta_factory = meta_factory
        self.train_frac = train_frac
        self.test_frac = test_frac
        self.embargo = embargo
        self._bases: list[Model] = []
        self._meta: object | None = None

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        tickers: pd.Series | None = None,
    ) -> StackingEnsemble:
        n = len(x)
        folds = walk_forward_splits(
            n,
            train_size=int(n * self.train_frac),
            test_size=max(1, int(n * self.test_frac)),
            embargo=self.embargo,
            expanding=True,
        )
        if not folds:
            msg = "dataset too small to build out-of-fold stacking features"
            raise ValueError(msg)

        n_bases = len(self.base_factories)
        oof = np.full((n, n_bases), np.nan)
        covered: set[int] = set()
        for fold in folds:
            tr, te = fold.train_idx, fold.test_idx
            t_tr = tickers.iloc[tr] if tickers is not None else None
            t_te = tickers.iloc[te] if tickers is not None else None
            w_tr = sample_weight.iloc[tr] if sample_weight is not None else None
            for j, factory in enumerate(self.base_factories):
                model = factory().fit(x.iloc[tr], y.iloc[tr], sample_weight=w_tr, tickers=t_tr)
                oof[te, j] = model.predict_proba(x.iloc[te], tickers=t_te)
            covered.update(te.tolist())

        idx = sorted(covered)
        x_meta = oof[idx, :]
        meta = self.meta_factory()
        meta.fit(x_meta, y.iloc[idx].to_numpy())  # ty: ignore[unresolved-attribute]
        self._meta = meta

        # Refit every base on all data for serving.
        self._bases = [
            factory().fit(x, y, sample_weight=sample_weight, tickers=tickers)
            for factory in self.base_factories
        ]
        return self

    def predict_proba(
        self,
        x: pd.DataFrame,
        *,
        tickers: pd.Series | None = None,
    ) -> np.ndarray:
        if self._meta is None or not self._bases:
            msg = "StackingEnsemble must be fit before predict_proba"
            raise RuntimeError(msg)
        cols = [np.asarray(b.predict_proba(x, tickers=tickers), dtype=float) for b in self._bases]
        stacked = np.column_stack(cols)
        proba = self._meta.predict_proba(stacked)  # ty: ignore[unresolved-attribute]
        return np.asarray(proba[:, 1], dtype=float)


class AveragingEnsemble:
    """Soft-vote ensemble: the (weighted) mean of its members' calibrated-scale win probabilities.

    Unlike :class:`StackingEnsemble` there is no meta-learner — just an average — which is robust on
    the thin per-asset data where a learned blender would itself overfit. Each member keeps its own
    feature subset: ``fit``/``predict_proba`` get the full feature frame and slice it per member, so
    members optimized on different feature groups compose without a column clash. Behaves as a
    single :class:`~berich.models.base.Model`, so it flows through ``oof_predict`` / the backtest /
    the promotion gate exactly like any other candidate.
    """

    def __init__(self, members: list[MemberSpec], *, weights: list[float] | None = None) -> None:
        if not members:
            msg = "AveragingEnsemble needs at least one member"
            raise ValueError(msg)
        if weights is not None and len(weights) != len(members):
            msg = "weights must match the number of members"
            raise ValueError(msg)
        self.members = members
        w = weights if weights is not None else [1.0] * len(members)
        total = float(sum(w))
        self.weights = [x / total for x in w] if total > 0 else [1.0 / len(w)] * len(w)
        self._fitted: list[Model] = []
        self._cols: list[list[str]] = []

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        *,
        tickers: pd.Series | None = None,
    ) -> AveragingEnsemble:
        self._fitted = []
        self._cols = []
        for factory, cols in self.members:
            use = [c for c in cols if c in x.columns] or list(x.columns)
            model = factory().fit(x[use], y, sample_weight=sample_weight, tickers=tickers)
            self._fitted.append(model)
            self._cols.append(use)
        return self

    def predict_proba(
        self,
        x: pd.DataFrame,
        *,
        tickers: pd.Series | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            msg = "AveragingEnsemble must be fit before predict_proba"
            raise RuntimeError(msg)
        acc = np.zeros(len(x), dtype=float)
        for model, cols, w in zip(self._fitted, self._cols, self.weights, strict=True):
            acc += w * np.asarray(model.predict_proba(x[cols], tickers=tickers), dtype=float)
        return acc


__all__ = ["AveragingEnsemble", "StackingEnsemble"]
