"""Tests for the interval-aware OhlcvStore (intraday POC).

The fatal blocker for intraday is ``.normalize()`` collapsing every 1h bar of a day
onto the same midnight index. The intraday store must keep the full timestamp; the
daily store must keep behaving exactly as before.
"""

from __future__ import annotations

import pandas as pd

from berich.data.store import OhlcvStore


def _hourly_frame(start: str, n: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1_000 + i for i in range(n)],
        },
        index=idx,
    )


def test_intraday_keeps_all_bars_of_one_day(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv_1h", interval="1h")
    store.save("BTC-USD", _hourly_frame("2024-01-02 00:00", 24))

    loaded = store.load("BTC-USD")
    # All 24 hourly bars survive — the regression guard against .normalize() collapse.
    assert len(loaded) == 24
    assert loaded.index[0] == pd.Timestamp("2024-01-02 00:00")
    assert loaded.index[-1] == pd.Timestamp("2024-01-02 23:00")


def test_daily_store_still_normalizes_to_midnight(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")  # default interval="1d"
    store.save("BTC-USD", _hourly_frame("2024-01-02 00:00", 24))

    loaded = store.load("BTC-USD")
    # Daily behavior unchanged: 24 same-day bars collapse to one midnight row.
    assert len(loaded) == 1
    assert loaded.index[0] == pd.Timestamp("2024-01-02 00:00")


def test_intraday_and_daily_use_separate_dirs(tmp_path):
    daily = OhlcvStore(tmp_path / "ohlcv")
    intraday = OhlcvStore(tmp_path / "ohlcv_1h", interval="1h")
    assert daily.ohlcv_dir != intraday.ohlcv_dir
    assert intraday._intraday is True
    assert daily._intraday is False
