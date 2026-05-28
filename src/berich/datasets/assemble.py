"""Turn cached OHLCV into a supervised, leakage-free training dataset.

`build_dataset` joins the causal feature matrix with forward-looking triple-barrier
labels, drops warm-up and incomplete-horizon rows, and returns a tidy
:class:`SupervisedDataset`. The binary target ``y`` is ``1`` when the upper barrier
was hit first (a winning swing long) and ``0`` otherwise, matching the
trend-probability objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from berich.features.build import (
    FEATURE_COLUMNS,
    HYG_TICKER,
    LQD_TICKER,
    MARKET_TICKER,
    SECTOR_MAP,
    TLT_TICKER,
    VIX9D_TICKER,
    VIX_TICKER,
    build_features,
)
from berich.labeling.triple_barrier import LabelConfig, triple_barrier_labels

if TYPE_CHECKING:
    from berich.data.store import OhlcvStore


CONTEXT_TICKERS: tuple[str, ...] = (
    VIX_TICKER,
    VIX9D_TICKER,
    TLT_TICKER,
    HYG_TICKER,
    LQD_TICKER,
    *sorted(set(SECTOR_MAP.values())),
)


def _load_context(store: OhlcvStore) -> dict[str, pd.DataFrame]:
    """Load every cross-asset series the feature builder might reference, if cached.

    Missing tickers are simply absent from the dict; the feature builder treats
    each cross-asset family as optional and emits NaN when the source isn't
    available, so a partial cache degrades gracefully instead of crashing.
    """
    out: dict[str, pd.DataFrame] = {}
    for ticker in CONTEXT_TICKERS:
        loaded = store.load(ticker)
        if loaded is not None and not loaded.empty:
            out[ticker] = loaded
    return out


@dataclass
class SupervisedDataset:
    """Aligned features/labels for one or more tickers, sorted by date."""

    x: pd.DataFrame  # rows = samples, cols = FEATURE_COLUMNS
    y: pd.Series  # binary target (1 = upper barrier hit first)
    weight: pd.Series  # sample weights (|realized return|)
    dates: pd.DatetimeIndex  # bar date per sample
    tickers: pd.Series  # ticker per sample

    def __len__(self) -> int:
        return len(self.x)


def build_ticker_dataset(
    df: pd.DataFrame,
    label_config: LabelConfig,
    *,
    ticker: str,
    market: pd.DataFrame | None = None,
    context: dict[str, pd.DataFrame] | None = None,
) -> SupervisedDataset:
    """Build a supervised dataset for a single OHLCV frame."""
    feats = build_features(df, market=market, context=context, ticker=ticker)
    labels = triple_barrier_labels(df, label_config)

    joined = feats.join(labels[["label", "sample_weight"]]).dropna()
    y = (joined["label"] == 1).astype(int)
    return SupervisedDataset(
        x=joined[FEATURE_COLUMNS],
        y=y,
        weight=joined["sample_weight"],
        dates=pd.DatetimeIndex(joined.index),
        tickers=pd.Series(ticker, index=joined.index),
    )


def build_dataset(
    store: OhlcvStore,
    tickers: list[str],
    label_config: LabelConfig,
) -> SupervisedDataset:
    """Build a combined dataset across tickers, sorted by date then ticker.

    Tickers that are absent from the cache are skipped. The result is globally
    date-sorted so walk-forward splits stay chronological across the panel. The
    market-regime ticker (SPY) and cross-asset context series (VIX, rates, credit,
    sector ETFs) are loaded once and broadcast to every ticker — with a one-bar
    lag enforced inside :func:`build_features` so cross-asset features can never
    use information from the same calendar day.
    """
    market = store.load(MARKET_TICKER)
    context = _load_context(store)
    parts = [
        build_ticker_dataset(df, label_config, ticker=t, market=market, context=context)
        for t in tickers
        if (df := store.load(t)) is not None and not df.empty
    ]
    if not parts:
        empty_idx = pd.DatetimeIndex([])
        return SupervisedDataset(
            x=pd.DataFrame(columns=pd.Index(FEATURE_COLUMNS)),
            y=pd.Series(dtype=int),
            weight=pd.Series(dtype=float),
            dates=empty_idx,
            tickers=pd.Series(dtype=str),
        )

    x = pd.concat([p.x for p in parts])
    y = pd.concat([p.y for p in parts])
    weight = pd.concat([p.weight for p in parts])
    tickers_s = pd.concat([p.tickers for p in parts])

    order = np.argsort(x.index.to_numpy(), kind="stable")
    return SupervisedDataset(
        x=x.iloc[order],
        y=y.iloc[order],
        weight=weight.iloc[order],
        dates=pd.DatetimeIndex(x.index[order]),
        tickers=tickers_s.iloc[order],
    )
