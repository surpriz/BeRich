"""Tests for the Parquet OHLCV store: schema normalization and incremental merge."""

from __future__ import annotations

import pandas as pd
import pytest

from berich.data.store import OHLCV_COLUMNS, OhlcvStore


def _frame(dates: list[str], close_start: float = 100.0) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    n = len(dates)
    return pd.DataFrame(
        {
            "open": [close_start + i for i in range(n)],
            "high": [close_start + i + 1 for i in range(n)],
            "low": [close_start + i - 1 for i in range(n)],
            "close": [close_start + i for i in range(n)],
            "volume": [1_000 + i for i in range(n)],
        },
        index=idx,
    )


def test_save_and_load_roundtrip(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")
    df = _frame(["2024-01-02", "2024-01-03"])
    store.save("aapl", df)

    loaded = store.load("AAPL")
    assert list(loaded.columns) == OHLCV_COLUMNS
    assert loaded.index.name == "date"
    assert len(loaded) == 2


def test_merge_dedupes_and_newer_wins(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save("X", _frame(["2024-01-02", "2024-01-03"], close_start=100.0))
    # Overlapping date 01-03 with a different value, plus a new date 01-04.
    store.save("X", _frame(["2024-01-03", "2024-01-04"], close_start=200.0))

    merged = store.load("X")
    assert len(merged) == 3  # 02, 03, 04 — no duplicate 03
    # Newer write wins on the overlap: 01-03 close came from the 200-based frame.
    assert merged.loc["2024-01-03", "close"] == 200.0


def test_last_date(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")
    assert store.last_date("NONE") is None
    store.save("X", _frame(["2024-01-02", "2024-01-05"]))
    assert store.last_date("X") == pd.Timestamp("2024-01-05")


def test_missing_columns_raise(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")
    bad = pd.DataFrame({"open": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(ValueError, match="missing columns"):
        store.save("X", bad)


def test_nan_close_bar_does_not_overwrite_good_close(tmp_path):
    # A provisional yfinance bar (volume present, NaN OHLC) must not blank a previously-good close.
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save("X", _frame(["2024-01-02", "2024-01-03"], close_start=100.0))
    nan_bar = _frame(["2024-01-03"], close_start=200.0)
    nan_bar.loc["2024-01-03", ["open", "high", "low", "close"]] = float("nan")
    store.save("X", nan_bar)

    merged = store.load("X")
    assert len(merged) == 2  # the NaN bar was dropped, not merged as a third row
    assert merged.loc["2024-01-03", "close"] == 101.0  # the good close survived
