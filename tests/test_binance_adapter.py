"""Tests for the Binance ccxt adapter — no network, ccxt client mocked."""

from __future__ import annotations

import sys
import types

import pandas as pd

from berich.data.binance_adapter import fetch_binance_ohlcv, to_binance_symbol
from berich.data.store import OHLCV_COLUMNS


def test_to_binance_symbol():
    assert to_binance_symbol("BTC-USD") == "BTC/USDT"
    assert to_binance_symbol("eth-usd") == "ETH/USDT"
    assert to_binance_symbol("SOL/USDT") == "SOL/USDT"  # passthrough


class _FakeBinance:
    """Returns two pages of 1h klines then stops (mimics ccxt pagination)."""

    _HOUR_MS = 3_600_000

    def __init__(self, *_args, **_kwargs):
        self._base = 1_700_000_000_000

    def milliseconds(self) -> int:
        return self._base + 4 * self._HOUR_MS  # 4 bars exist

    def fetch_ohlcv(self, _symbol, *, timeframe, since, limit):  # noqa: ARG002 — ccxt signature
        rows = []
        ts = max(since, self._base)
        while ts < self.milliseconds() and len(rows) < limit:
            i = (ts - self._base) // self._HOUR_MS
            rows.append([ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0 + i])
            ts += self._HOUR_MS
        return rows


def _install_fake_ccxt(monkeypatch):
    fake = types.ModuleType("ccxt")
    fake.binance = _FakeBinance
    monkeypatch.setitem(sys.modules, "ccxt", fake)


def test_fetch_schema_and_distinct_timestamps(monkeypatch):
    _install_fake_ccxt(monkeypatch)
    df = fetch_binance_ohlcv(
        "BTC/USDT", start=pd.Timestamp("2023-11-14"), interval="1h", limit_per_call=2
    )

    assert list(df.columns) == OHLCV_COLUMNS
    assert df.index.name == "date"
    assert df.index.tz is None
    # Pagination across two pages of 2 returned all 4 distinct hourly bars.
    assert len(df) == 4
    assert df.index.is_unique
    # Distinct intraday timestamps preserved (not collapsed to a single day).
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_fetch_empty_returns_canonical_empty(monkeypatch):
    fake = types.ModuleType("ccxt")

    class _Empty(_FakeBinance):
        def fetch_ohlcv(self, *_a, **_k):
            return []

    fake.binance = _Empty
    monkeypatch.setitem(sys.modules, "ccxt", fake)

    df = fetch_binance_ohlcv("BTC/USDT", start=pd.Timestamp("2023-11-14"))
    assert df.empty
    assert list(df.columns) == OHLCV_COLUMNS
