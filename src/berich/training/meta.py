"""Build and train the meta-labeling dataset (leak-free, via out-of-fold primary probas).

The meta dataset restricts to bars where the primary model said BUY (out-of-fold proba
>= threshold), attaches that out-of-fold proba as an extra feature, and labels each row by
whether the primary BUY would have won (the binary triple-barrier outcome). Training on the
*out-of-fold* primary proba — never the in-sample one — is the single correctness-critical
detail: it ensures the meta model never sees a leaked primary prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.metrics import roc_auc_score

from berich.models.meta_labeler import PRIMARY_PROBA_COL, MetaLabeler

if TYPE_CHECKING:
    from berich.datasets.assemble import SupervisedDataset
    from berich.training.walk_forward import OofResult


@dataclass
class MetaDataset:
    """Features (+ primary proba) and the act-or-not meta label."""

    x: pd.DataFrame  # base features + PRIMARY_PROBA_COL
    y: pd.Series  # 1 if the primary BUY would have won, else 0
    base_features: list[str]

    def __len__(self) -> int:
        return len(self.x)


def build_meta_dataset(
    base: SupervisedDataset,
    oof: OofResult,
    *,
    buy_threshold: float,
) -> MetaDataset:
    """Assemble the meta dataset from base features and out-of-fold primary probas.

    Aligns base rows to OOF rows on ``(date, ticker)``, keeps only BUY candidates
    (oof proba >= ``buy_threshold``), and adds the OOF proba as ``primary_proba``.
    """
    base_features = list(base.x.columns)
    base_df = base.x.copy()
    base_df["__date"] = base.dates
    base_df["__ticker"] = base.tickers.to_numpy()
    base_df["__y"] = base.y.to_numpy()

    oof_df = oof.frame.reset_index().rename(columns={"date": "__date", "ticker": "__ticker"})
    oof_df = oof_df[["__date", "__ticker", "proba", "y_true"]]

    merged = base_df.merge(oof_df, on=["__date", "__ticker"], how="inner")
    buys = merged[merged["proba"] >= buy_threshold].copy()

    x = buys[base_features].copy()
    x[PRIMARY_PROBA_COL] = buys["proba"].to_numpy()
    y = pd.Series(buys["y_true"].to_numpy().astype(int), name="acted_win")
    return MetaDataset(
        x=x.reset_index(drop=True), y=y.reset_index(drop=True), base_features=base_features
    )


def train_meta_model(meta: MetaDataset) -> tuple[MetaLabeler, float]:
    """Fit a :class:`MetaLabeler` on the meta dataset; return (model, in-sample AUC).

    The leak guard lives upstream (the primary proba is out-of-fold). The returned AUC is
    in-sample and for logging only — promotion of the meta filter is evaluated by its
    effect on the paper-trade precision, not this number.
    """
    model = MetaLabeler(meta.base_features).fit(meta.x, meta.y)
    auc = float("nan")
    if meta.y.nunique() >= 2:  # noqa: PLR2004
        auc = float(roc_auc_score(meta.y, model.predict_proba(meta.x)))
    return model, auc


__all__ = ["MetaDataset", "build_meta_dataset", "train_meta_model"]
