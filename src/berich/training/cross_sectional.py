"""Walk-forward out-of-sample scoring for the cross-sectional long/short track.

Near-identical to :func:`berich.training.walk_forward.oof_predict` — it reuses the same
leak-free :func:`walk_forward_splits` seam — but the target is a continuous cross-sectional
score, so the diagnostic is the **rank information coefficient** (mean per-date Spearman
correlation between predicted score and realized residual) instead of ROC-AUC, and the OOF
column is ``score`` rather than ``proba``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from berich.datasets.splits import walk_forward_splits

if TYPE_CHECKING:
    from berich.datasets.cross_sectional import PanelDataset
    from berich.models.base import Model

ModelFactory = Callable[[], "Model"]


@dataclass
class CrossSectionalOof:
    """Out-of-sample cross-sectional scores aligned to the panel's test rows."""

    frame: pd.DataFrame  # columns: score, y_true, ticker; index = bar date

    def _daily_ic(self) -> pd.Series:
        """Per-date Spearman rank correlation between score and realized residual."""

        def _ic(g: pd.DataFrame) -> float:
            if len(g) < 2 or g["score"].nunique() < 2 or g["y_true"].nunique() < 2:  # noqa: PLR2004
                return float("nan")
            return float(g["score"].corr(g["y_true"], method="spearman"))

        ic = self.frame.groupby(level=0)[["score", "y_true"]].apply(_ic)
        return pd.Series(ic, dtype=float).dropna()

    @property
    def rank_ic(self) -> float:
        """Mean daily rank IC (0 == no cross-sectional ranking skill)."""
        ic = self._daily_ic()
        return float(ic.mean()) if not ic.empty else float("nan")

    @property
    def ic_t_stat(self) -> float:
        """t-stat of the daily IC series (information ratio of the IC)."""
        ic = self._daily_ic()
        if len(ic) < 2 or ic.std(ddof=1) == 0:  # noqa: PLR2004
            return float("nan")
        return float(ic.mean() / ic.std(ddof=1) * math.sqrt(len(ic)))


def oof_predict_cross_sectional(
    dataset: PanelDataset,
    model_factory: ModelFactory,
    *,
    train_frac: float = 0.5,
    test_frac: float = 0.1,
    embargo: int = 5,
) -> CrossSectionalOof:
    """Run walk-forward training and collect out-of-sample cross-sectional scores."""
    n = len(dataset)
    folds = walk_forward_splits(
        n,
        train_size=int(n * train_frac),
        test_size=max(1, int(n * test_frac)),
        embargo=embargo,
        expanding=True,
    )
    if not folds:
        msg = "panel too small for the requested walk-forward configuration"
        raise ValueError(msg)

    rows: list[pd.DataFrame] = []
    for fold in folds:
        model = model_factory()
        model.fit(
            dataset.x.iloc[fold.train_idx],
            dataset.y.iloc[fold.train_idx],
            sample_weight=dataset.weight.iloc[fold.train_idx],
            tickers=dataset.tickers.iloc[fold.train_idx],
        )
        x_te = dataset.x.iloc[fold.test_idx]
        t_te = dataset.tickers.iloc[fold.test_idx]
        score = model.predict_proba(x_te, tickers=t_te)
        rows.append(
            pd.DataFrame(
                {
                    "score": np.asarray(score, dtype=float),
                    "y_true": dataset.y.iloc[fold.test_idx].to_numpy(),
                    "ticker": dataset.tickers.iloc[fold.test_idx].to_numpy(),
                },
                index=dataset.dates[fold.test_idx],
            )
        )
    frame = pd.concat(rows)
    frame.index.name = "date"
    return CrossSectionalOof(frame=frame.sort_index())


__all__ = ["CrossSectionalOof", "oof_predict_cross_sectional"]
