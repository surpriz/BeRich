"""Cross-sectional panel assembly for the market-neutral long/short track.

Mirrors :func:`berich.datasets.assemble.build_dataset` but produces a *continuous*
cross-sectional target instead of a binary triple-barrier label: per ticker we attach
the beta-residualized forward return, then standardize it **within each date** (z-score
or rank percentile) so every date is on the same scale and the pooled walk-forward OOF
is comparable. The resulting :class:`PanelDataset` is structurally identical to
:class:`SupervisedDataset` (same five fields) so it is drop-in for the cross-sectional
walk-forward in :mod:`berich.training.cross_sectional`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from berich.features.build import FEATURE_COLUMNS, build_features
from berich.labeling.cross_sectional import CrossSectionalLabelConfig, forward_return_labels

if TYPE_CHECKING:
    from berich.data.store import OhlcvStore


@dataclass
class PanelDataset:
    """Aligned features / cross-sectional target for a panel of tickers, date-sorted."""

    x: pd.DataFrame  # rows = samples, cols = FEATURE_COLUMNS
    y: pd.Series  # continuous cross-sectional target (z-score or rank of residual)
    weight: pd.Series  # sample weights (|residual|)
    dates: pd.DatetimeIndex  # bar date per sample
    tickers: pd.Series  # ticker per sample

    def __len__(self) -> int:
        return len(self.x)


def _standardize(resid: pd.Series, dates: pd.DatetimeIndex, method: str) -> pd.Series:
    """Standardize the residual within each date (z-score or rank percentile)."""
    grouped = resid.groupby(dates)
    if method == "rank":
        return grouped.transform(lambda s: s.rank(pct=True) - 0.5)
    return grouped.transform(lambda s: (s - s.mean()) / s.std(ddof=0))


def build_panel_dataset(
    store: OhlcvStore,
    tickers: list[str],
    label_config: CrossSectionalLabelConfig,
    *,
    market_ticker: str = "SPY",
    min_names_per_date: int = 20,
) -> PanelDataset:
    """Build a date-sorted cross-sectional panel with a within-date standardized target.

    Tickers absent from the cache are skipped. Dates with fewer than
    ``min_names_per_date`` names (too thin to rank into deciles) are dropped.
    """
    market = store.load(market_ticker)

    parts: list[pd.DataFrame] = []
    for t in tickers:
        df = store.load(t)
        if df is None or df.empty:
            continue
        feats = build_features(df, market=market)
        labels = forward_return_labels(df, label_config, market=market)
        joined = feats[FEATURE_COLUMNS].join(labels[["resid", "sample_weight"]]).dropna()
        if joined.empty:
            continue
        joined = joined.assign(ticker=t)
        parts.append(joined)

    cols = list(FEATURE_COLUMNS)
    if not parts:
        return PanelDataset(
            x=pd.DataFrame(columns=pd.Index(cols)),
            y=pd.Series(dtype=float),
            weight=pd.Series(dtype=float),
            dates=pd.DatetimeIndex([]),
            tickers=pd.Series(dtype=str),
        )

    panel = pd.concat(parts)
    order = np.argsort(panel.index.to_numpy(), kind="stable")
    panel = panel.iloc[order]
    dates = pd.DatetimeIndex(panel.index)

    # Drop thin cross-sections, then standardize the residual within each remaining date.
    counts = panel.groupby(level=0)["resid"].transform("size")
    panel = panel[counts >= min_names_per_date]
    dates = pd.DatetimeIndex(panel.index)

    y = _standardize(panel["resid"], dates, label_config.standardize)
    keep = y.notna().to_numpy()
    panel = panel[keep]
    y = y[keep]
    dates = pd.DatetimeIndex(panel.index)

    return PanelDataset(
        x=panel[cols],
        y=pd.Series(y.to_numpy(), index=panel.index, name="y"),
        weight=panel["sample_weight"],
        dates=dates,
        tickers=panel["ticker"],
    )


__all__ = ["PanelDataset", "build_panel_dataset"]
