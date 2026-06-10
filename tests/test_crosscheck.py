"""Tests for the Stooq second-source cross-check (no network: stooq_closes is patched)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.data import crosscheck
from berich.data.crosscheck import cross_check_ticker, stooq_symbol
from berich.data.store import OhlcvStore


def test_stooq_symbol_mapping():
    assert stooq_symbol("AAPL") == "aapl.us"
    assert stooq_symbol("EURUSD=X") == "eurusd"
    # No reliable mapping: crypto, FR listings, futures, indices.
    assert stooq_symbol("BTC-USD") is None
    assert stooq_symbol("AIR.PA") is None
    assert stooq_symbol("GC=F") is None
    assert stooq_symbol("^GSPC") is None


def _store_with(tmp_path, ticker: str, closes: pd.Series) -> OhlcvStore:
    store = OhlcvStore(tmp_path / "ohlcv")
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "volume": 1_000_000.0,
        },
        index=closes.index,
    )
    df.index.name = "date"
    store.save(ticker, df)
    return store


def test_cross_check_flags_recent_disagreement(tmp_path, monkeypatch):
    idx = pd.bdate_range("2024-01-02", periods=10)
    ours = pd.Series(np.full(10, 100.0), index=idx)
    store = _store_with(tmp_path, "AAPL", ours)
    # Stooq agrees everywhere except the last bar, which is off by 5%.
    theirs = ours.copy()
    theirs.iloc[-1] = 105.0
    monkeypatch.setattr(crosscheck, "stooq_closes", lambda _symbol, **_kw: theirs)

    findings = cross_check_ticker(store, "AAPL")
    assert len(findings) == 1
    assert findings[0]["date"] == idx[-1].date().isoformat()
    assert findings[0]["rel_diff"] > 0.01


def test_cross_check_quiet_when_sources_agree(tmp_path, monkeypatch):
    idx = pd.bdate_range("2024-01-02", periods=10)
    ours = pd.Series(np.linspace(100.0, 110.0, 10), index=idx)
    store = _store_with(tmp_path, "MSFT", ours)
    monkeypatch.setattr(crosscheck, "stooq_closes", lambda _symbol, **_kw: ours.copy())
    assert cross_check_ticker(store, "MSFT") == []


def test_cross_check_skips_unmappable_and_network_failures(tmp_path, monkeypatch):
    idx = pd.bdate_range("2024-01-02", periods=10)
    ours = pd.Series(np.full(10, 100.0), index=idx)
    store = _store_with(tmp_path, "AAPL", ours)
    # Crypto: no mapping, no fetch attempted.
    assert cross_check_ticker(store, "BTC-USD") == []

    # Stooq down: logged, never raised, never a false alarm.
    def _boom(_symbol, **_kw):
        msg = "stooq unreachable"
        raise OSError(msg)

    monkeypatch.setattr(crosscheck, "stooq_closes", _boom)
    assert cross_check_ticker(store, "AAPL") == []
