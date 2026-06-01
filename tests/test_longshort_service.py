"""Tests for live long/short basket generation, persistence, and paper equity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.config import Config, LongShortConfigModel
from berich.signals.longshort_service import (
    LongShortStore,
    generate_longshort_book,
    longshort_equity,
)


def _ohlcv(seed: int, n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e6, 5e6, n).astype(float),
        },
        index=idx,
    )


class _FakeStore:
    def __init__(self, frames):
        self._frames = frames

    def load(self, ticker):
        return self._frames.get(ticker)


def _config(tmp_path) -> tuple[Config, _FakeStore]:
    tickers = [f"T{i:02d}" for i in range(12)]
    frames = {t: _ohlcv(i) for i, t in enumerate(tickers)}
    frames["SPY"] = _ohlcv(99)
    config = Config(
        data_dir=tmp_path,
        watchlist=tickers,
        longshort=LongShortConfigModel(universe="all", min_names_per_date=5, horizon_days=5),
    )
    return config, _FakeStore(frames)


def test_generate_longshort_book_is_dollar_neutral(tmp_path):
    config, store = _config(tmp_path)
    book = generate_longshort_book(config, store)
    assert book is not None
    assert book.positions
    assert {p.side for p in book.positions} <= {"LONG", "SHORT"}
    # Dollar-neutral: long gross ~ short gross.
    long_w = sum(p.weight for p in book.positions if p.side == "LONG")
    short_w = sum(-p.weight for p in book.positions if p.side == "SHORT")
    assert abs(long_w - short_w) < 0.2


def test_longshort_store_roundtrip_and_equity(tmp_path):
    config, store = _config(tmp_path)
    book = generate_longshort_book(config, store)
    saved = LongShortStore(config.db_path).save(book)
    assert saved == len(book.positions)
    latest = LongShortStore(config.db_path).latest()
    assert len(latest) == len(book.positions)

    summary = longshort_equity(config, store)
    assert summary["n_baskets"] >= 1
