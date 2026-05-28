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
    MARKET_TICKER,
    build_features,
    feature_columns,
)
from berich.labeling.triple_barrier import LabelConfig, triple_barrier_labels

if TYPE_CHECKING:
    from berich.data.earnings import EarningsStore
    from berich.data.news import NewsStore
    from berich.data.store import OhlcvStore


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
    earnings: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
) -> SupervisedDataset:
    """Build a supervised dataset for a single OHLCV frame."""
    feats = build_features(df, market=market, earnings=earnings, news=news)
    labels = triple_barrier_labels(df, label_config)

    joined = feats.join(labels[["label", "sample_weight"]]).dropna()
    y = (joined["label"] == 1).astype(int)
    cols = feature_columns(earnings=earnings is not None, news=news is not None)
    return SupervisedDataset(
        x=joined[cols],
        y=y,
        weight=joined["sample_weight"],
        dates=pd.DatetimeIndex(joined.index),
        tickers=pd.Series(ticker, index=joined.index),
    )


def build_dataset(
    store: OhlcvStore,
    tickers: list[str],
    label_config: LabelConfig,
    *,
    earnings_store: EarningsStore | None = None,
    news_store: NewsStore | None = None,
) -> SupervisedDataset:
    """Build a combined dataset across tickers, sorted by date then ticker.

    Tickers absent from the cache are skipped. The result is date-sorted so
    walk-forward splits stay chronological across the panel. SPY is loaded
    once and broadcast as the market regime; ``earnings_store`` and
    ``news_store`` add their respective feature columns when supplied.
    Neutral defaults inside the feature builders keep tickers without an
    earnings track (SPY) or with no news cached (or empty) in the panel
    rather than silently dropping them at ``dropna``.

    The model registry's metadata records which mode each artifact was
    trained with so serving stays in sync at load time.
    """
    market = store.load(MARKET_TICKER)
    use_earnings = earnings_store is not None
    use_news = news_store is not None

    def _earnings_for(ticker: str) -> pd.DataFrame | None:
        if earnings_store is None:
            return None
        loaded = earnings_store.load(ticker)
        return loaded if loaded is not None else pd.DataFrame()

    def _news_for(ticker: str) -> pd.DataFrame | None:
        if news_store is None:
            return None
        loaded = news_store.load(ticker)
        return loaded if loaded is not None else pd.DataFrame()

    parts = [
        build_ticker_dataset(
            df,
            label_config,
            ticker=t,
            market=market,
            earnings=_earnings_for(t) if use_earnings else None,
            news=_news_for(t) if use_news else None,
        )
        for t in tickers
        if (df := store.load(t)) is not None and not df.empty
    ]
    if not parts:
        empty_idx = pd.DatetimeIndex([])
        return SupervisedDataset(
            x=pd.DataFrame(columns=pd.Index(feature_columns(earnings=use_earnings, news=use_news))),
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
