"""Walk-forward out-of-sample prediction.

Trains a fresh model on each fold's train window and predicts the test block,
concatenating the test-block predictions into a single out-of-sample (OOS) series.
These OOS probabilities — never seen by the model during training — are what the
backtester and metrics consume. Anything computed on in-sample data would be
optimistic to the point of useless.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from berich.datasets.splits import walk_forward_splits

if TYPE_CHECKING:
    from berich.datasets.assemble import SupervisedDataset
    from berich.models.base import Model

ModelFactory = Callable[[], "Model"]


@dataclass
class OofResult:
    """Out-of-sample predictions aligned to the dataset's test rows."""

    frame: pd.DataFrame  # columns: proba, y_true, ticker; index = bar date

    @property
    def auc(self) -> float:
        """ROC-AUC over all OOS predictions (0.5 == coin flip)."""
        y = self.frame["y_true"]
        if y.nunique() < 2:  # noqa: PLR2004
            return float("nan")
        return float(roc_auc_score(y, self.frame["proba"]))


def oof_predict(
    dataset: SupervisedDataset,
    model_factory: ModelFactory,
    *,
    train_frac: float = 0.5,
    test_frac: float = 0.1,
    embargo: int = 10,
    fold_callback: Callable[[int, pd.DataFrame], None] | None = None,
) -> OofResult:
    """Run walk-forward training and collect out-of-sample probabilities.

    Args:
        dataset: globally date-sorted supervised data.
        model_factory: zero-arg callable returning a fresh, untrained model.
        train_frac: initial train window as a fraction of all samples.
        test_frac: test block size as a fraction of all samples.
        embargo: rows skipped between train and test (>= label horizon).
        fold_callback: called after each non-final fold with ``(fold_index,
            accumulated_oof_frame)``. An HPO caller can report the partial score and raise
            (e.g. ``optuna.TrialPruned``) to abort a hopeless trial early — the exception
            propagates. Walk-forward itself stays Optuna-agnostic.
    """
    n = len(dataset)
    folds = walk_forward_splits(
        n,
        train_size=int(n * train_frac),
        test_size=max(1, int(n * test_frac)),
        embargo=embargo,
        expanding=True,
    )
    if not folds:
        msg = "dataset too small for the requested walk-forward configuration"
        raise ValueError(msg)

    rows: list[pd.DataFrame] = []
    for fold_idx, fold in enumerate(folds):
        model = model_factory()
        x_tr = dataset.x.iloc[fold.train_idx]
        y_tr = dataset.y.iloc[fold.train_idx]
        w_tr = dataset.weight.iloc[fold.train_idx]
        t_tr = dataset.tickers.iloc[fold.train_idx]
        model.fit(x_tr, y_tr, sample_weight=w_tr, tickers=t_tr)

        x_te = dataset.x.iloc[fold.test_idx]
        t_te = dataset.tickers.iloc[fold.test_idx]
        proba = model.predict_proba(x_te, tickers=t_te)
        rows.append(
            pd.DataFrame(
                {
                    "proba": proba,
                    "y_true": dataset.y.iloc[fold.test_idx].to_numpy(),
                    "ticker": dataset.tickers.iloc[fold.test_idx].to_numpy(),
                },
                index=dataset.dates[fold.test_idx],
            )
        )
        if fold_callback is not None and fold_idx < len(folds) - 1:
            fold_callback(fold_idx, pd.concat(rows))
    frame = pd.concat(rows)
    frame.index.name = "date"
    return OofResult(frame=frame.sort_index())


def summarize_importances(model: Model, feature_names: list[str]) -> pd.Series:
    """Return feature importances as a sorted Series, if the model exposes them."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.Series(dtype=float)
    return pd.Series(np.asarray(importances), index=feature_names).sort_values(ascending=False)
