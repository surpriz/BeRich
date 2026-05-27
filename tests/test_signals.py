"""Tests for signal classification, position sizing, and DuckDB persistence."""

from __future__ import annotations

import pandas as pd

from berich.config import Config, SignalConfig
from berich.signals.service import BUY, NEUTRAL, SELL, Signal, _classify, _size_position
from berich.signals.store import SignalStore


def _config() -> Config:
    return Config(
        signals=SignalConfig(
            buy_threshold=0.55, sell_threshold=0.30, capital=10_000.0, risk_pct=0.01
        )
    )


def test_classify_thresholds():
    cfg = _config()
    assert _classify(0.60, cfg) == BUY
    assert _classify(0.45, cfg) == NEUTRAL
    assert _classify(0.20, cfg) == SELL


def test_size_position_risk_based():
    cfg = _config()
    # Risk 1% of 10k = $100; stop distance $5 -> 20 shares, $2000 notional.
    shares, notional = _size_position(entry=100.0, stop=95.0, config=cfg)
    assert shares == 20
    assert notional == 2000.0


def test_size_position_rejects_nonpositive_stop_distance():
    cfg = _config()
    shares, notional = _size_position(entry=100.0, stop=100.0, config=cfg)
    assert shares == 0
    assert notional == 0.0


def _signal(ticker: str, proba: float, date: str = "2024-01-05") -> Signal:
    return Signal(
        date=pd.Timestamp(date),
        ticker=ticker,
        signal=BUY,
        proba=proba,
        entry=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        size_shares=20,
        notional=2000.0,
    )


def test_signal_store_roundtrip_and_upsert(tmp_path):
    store = SignalStore(tmp_path / "berich.duckdb")
    store.save([_signal("AAPL", 0.6), _signal("MSFT", 0.4)])

    latest = store.latest()
    assert len(latest) == 2
    # Ordered by proba descending.
    assert latest.iloc[0]["ticker"] == "AAPL"

    # Re-saving the same (date, ticker) overwrites instead of duplicating.
    store.save([_signal("AAPL", 0.7)])
    latest = store.latest()
    assert len(latest) == 2
    aapl = latest[latest["ticker"] == "AAPL"].iloc[0]
    assert abs(aapl["proba"] - 0.7) < 1e-9


def test_signal_store_history(tmp_path):
    store = SignalStore(tmp_path / "berich.duckdb")
    store.save([_signal("AAPL", 0.6, "2024-01-04"), _signal("AAPL", 0.5, "2024-01-05")])
    hist = store.history("AAPL")
    assert len(hist) == 2
    assert list(hist["date"]) == sorted(hist["date"])
