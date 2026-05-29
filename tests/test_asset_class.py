"""Tests for asset-class market-reference resolution and build_dataset threading."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.datasets.assemble import build_dataset
from berich.features.build import MARKET_TICKER, market_reference_for
from berich.labeling.triple_barrier import LabelConfig


def test_market_reference_defaults_and_overrides():
    assert market_reference_for("crypto") == "BTC-USD"
    assert market_reference_for("us_stocks") == MARKET_TICKER
    assert market_reference_for("unknown_class") == MARKET_TICKER  # safe fallback


def _ohlcv(seed: int, n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
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


def test_build_dataset_uses_custom_market_ticker():
    # A non-SPY market reference is honored; using a present ticker yields a non-empty set.
    store = _FakeStore({"BTC-USD": _ohlcv(1), "ETH-USD": _ohlcv(2)})
    ds = build_dataset(store, ["ETH-USD"], LabelConfig(), market_ticker="BTC-USD")
    assert len(ds) > 0
    assert list(ds.x.columns)  # FEATURE_COLUMNS present
