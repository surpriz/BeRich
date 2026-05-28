"""Assemble the PEAD (event-level) supervised dataset.

Mirrors :mod:`berich.datasets.assemble` but at the event grain: one row per
``(ticker, earnings event)`` instead of one per trading day. The dataset is
sorted by ``entry_date`` (the trading day used as the long entry) so
walk-forward folds stay strictly chronological even across tickers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from berich.features.build import MARKET_TICKER, SECTOR_MAP
from berich.features.pead_features import PEAD_FEATURE_COLUMNS, build_pead_features
from berich.labeling.pead import build_pead_events

if TYPE_CHECKING:
    from berich.data.earnings import EarningsStore
    from berich.data.news import NewsStore
    from berich.data.store import OhlcvStore


@dataclass
class PeadDataset:
    """Event-level supervised data, sorted by entry_date for walk-forward training."""

    events: pd.DataFrame  # one row per event with ticker/event_date/entry_date/labels
    x: pd.DataFrame  # features aligned to events
    y: pd.Series  # binary target — drift_5d by default
    entry_dates: pd.DatetimeIndex
    tickers: pd.Series

    def __len__(self) -> int:
        return len(self.x)


def build_pead_dataset(  # noqa: C901 — assembly fan-out across data sources is intrinsic
    store: OhlcvStore,
    earnings_store: EarningsStore,
    tickers: list[str],
    *,
    news_store: NewsStore | None = None,
    label_horizon: str = "5d",
) -> PeadDataset:
    """Build the event-level PEAD dataset across ``tickers``.

    ``label_horizon`` selects the target column: ``"5d"`` → ``label_drift_5d``,
    ``"20d"`` → ``label_drift_20d``. Tickers without OHLCV or without an
    earnings calendar are silently skipped (they contribute no events).
    """
    if label_horizon not in {"5d", "20d"}:
        msg = f"label_horizon must be '5d' or '20d', got {label_horizon!r}"
        raise ValueError(msg)

    market = store.load(MARKET_TICKER)
    ohlcv_by_ticker: dict[str, pd.DataFrame] = {}
    earnings_by_ticker: dict[str, pd.DataFrame] = {}
    news_by_ticker: dict[str, pd.DataFrame] = {}

    event_parts: list[pd.DataFrame] = []
    for ticker in tickers:
        ohlcv = store.load(ticker)
        if ohlcv is None or ohlcv.empty:
            continue
        ohlcv_by_ticker[ticker] = ohlcv
        earnings = earnings_store.load(ticker)
        if earnings is None or earnings.empty:
            continue
        earnings_by_ticker[ticker] = earnings
        if news_store is not None:
            news = news_store.load(ticker)
            if news is not None and not news.empty:
                # Only rows with a usable timestamp are worth the lookup cost.
                news = news.dropna(subset=["time_published"])
                if not news.empty:
                    news_by_ticker[ticker] = news
        ticker_events = build_pead_events(ohlcv, earnings, ticker=ticker)
        if not ticker_events.empty:
            event_parts.append(ticker_events)

    if not event_parts:
        return _empty_dataset()

    events = pd.concat(event_parts, ignore_index=True)
    events = events.sort_values("entry_date", kind="stable").reset_index(drop=True)

    # Pre-load sector ETFs that are referenced by any ticker in this batch.
    sector_etf_codes: set[str] = {
        s for t in earnings_by_ticker if (s := SECTOR_MAP.get(t.upper())) is not None
    }
    sector_by_etf: dict[str, pd.DataFrame] = {}
    for etf in sector_etf_codes:
        df = store.load(etf)
        if df is not None and not df.empty:
            sector_by_etf[etf] = df

    features = build_pead_features(
        events,
        ohlcv_by_ticker=ohlcv_by_ticker,
        earnings_by_ticker=earnings_by_ticker,
        market=market,
        news_by_ticker=news_by_ticker or None,
        sector_by_etf=sector_by_etf or None,
    )

    label_col = "label_drift_5d" if label_horizon == "5d" else "label_drift_20d"
    y = events[label_col].astype(int)
    return PeadDataset(
        events=events,
        x=features,
        y=y,
        entry_dates=pd.DatetimeIndex(events["entry_date"]),
        tickers=events["ticker"].astype(str),
    )


def _empty_dataset() -> PeadDataset:
    empty_idx = pd.DatetimeIndex([])
    return PeadDataset(
        events=pd.DataFrame(),
        x=pd.DataFrame(columns=pd.Index(PEAD_FEATURE_COLUMNS)),
        y=pd.Series(dtype=int),
        entry_dates=empty_idx,
        tickers=pd.Series(dtype=str),
    )


def split_walk_forward(
    dataset: PeadDataset,
    *,
    n_folds: int = 5,
    min_train: int = 500,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological event-level walk-forward splits.

    Returns a list of ``(train_idx, test_idx)`` integer-position pairs. The
    first fold uses the first ``min_train`` events; each subsequent fold
    adds an equal slice of the remaining events to the test set and to the
    next fold's training set. Empty if the dataset is too small.
    """
    n = len(dataset)
    if n <= min_train:
        return []
    remaining = n - min_train
    fold_size = max(1, remaining // n_folds)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train
    while train_end < n:
        test_end = min(train_end + fold_size, n)
        folds.append((np.arange(train_end), np.arange(train_end, test_end)))
        train_end = test_end
    return folds
