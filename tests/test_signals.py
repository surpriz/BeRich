"""Tests for signal classification, position sizing, and DuckDB persistence."""

from __future__ import annotations

import pandas as pd
import pytest

from berich.config import Config, SignalConfig
from berich.signals.service import (
    BUY,
    LONG,
    NEUTRAL,
    SELL,
    SHORT,
    Signal,
    _classify,
    _decide,
    _expected_return,
    _price_decimals,
    _size_position,
)
from berich.signals.store import SignalStore


def _config() -> Config:
    return Config(
        signals=SignalConfig(
            buy_threshold=0.55, sell_threshold=0.30, capital=10_000.0, risk_pct=0.01
        )
    )


def _ls_config(*, enable_short: bool = True) -> Config:
    return Config(
        signals=SignalConfig(
            buy_threshold=0.55,
            short_threshold=0.55,
            enable_short=enable_short,
            capital=10_000.0,
            risk_pct=0.01,
        )
    )


def test_classify_thresholds():
    cfg = _config()
    assert _classify(0.60, cfg) == BUY
    assert _classify(0.45, cfg) == NEUTRAL
    assert _classify(0.20, cfg) == SELL


def test_decide_picks_long():
    assert _decide(0.60, 0.20, _ls_config()) == (LONG, "long")


def test_decide_picks_short():
    assert _decide(0.20, 0.60, _ls_config()) == (SHORT, "short")


def test_decide_neutral_when_both_below_thresholds():
    assert _decide(0.40, 0.40, _ls_config()) == (NEUTRAL, "long")


def test_decide_tie_favors_long():
    # Both clear their threshold and are equal -> deterministic long bias.
    assert _decide(0.60, 0.60, _ls_config()) == (LONG, "long")


def test_decide_no_short_model_never_shorts():
    assert _decide(0.60, None, _ls_config()) == (LONG, "long")
    assert _decide(0.40, None, _ls_config()) == (NEUTRAL, "long")


def test_decide_short_suppressed_when_disabled():
    # A strong short is ignored when enable_short is False.
    assert _decide(0.20, 0.90, _ls_config(enable_short=False)) == (NEUTRAL, "long")


def test_size_position_risk_based():
    cfg = _config()
    # Risk 1% of 10k = $100; stop distance $5 -> 20 shares, $2000 notional.
    shares, notional = _size_position(entry=100.0, stop=95.0, config=cfg)
    assert shares == 20
    assert notional == 2000.0


def test_size_position_uses_absolute_distance():
    # A short's stop sits ABOVE entry; |100 - 105| == 5 must size like the long case.
    cfg = _config()
    assert _size_position(entry=100.0, stop=105.0, config=cfg) == _size_position(
        entry=100.0, stop=95.0, config=cfg
    )


def test_size_position_capped_at_capital_no_leverage():
    # A low-priced FX pair with a tiny stop distance would risk-size to tens of thousands
    # of units; the no-leverage cap keeps notional <= capital.
    cfg = _config()  # capital 10k, risk 1%
    shares, notional = _size_position(entry=1.1664, stop=1.1602, config=cfg)
    assert notional <= cfg.signals.capital
    assert shares == int(cfg.signals.capital // 1.1664)


def test_expected_return_triple_barrier_expectancy():
    # Long: entry 100, target 110 (+10% reward), stop 95 (-5% risk), P(win)=0.4.
    # E = 0.4*0.10 - 0.6*0.05 = 0.04 - 0.03 = +0.01.
    er = _expected_return(0.4, entry=100.0, stop=95.0, target=110.0)
    assert er == pytest.approx(0.01)


def test_expected_return_direction_agnostic_via_abs():
    # Short: entry 100, target 90 (reward), stop 105 (risk). Same |distances| as a 10/5 long
    # scaled, so the abs-based formula yields P(win)*0.10 - P(loss)*0.05.
    er = _expected_return(0.4, entry=100.0, stop=105.0, target=90.0)
    assert er == pytest.approx(0.4 * 0.10 - 0.6 * 0.05)


def test_price_decimals_scales_with_magnitude():
    assert _price_decimals(212.5) == 2  # equity
    assert _price_decimals(1.1664) == 4  # FX pair
    assert _price_decimals(0.42) == 6  # sub-unit (e.g. some crypto/penny)


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


def test_signal_store_roundtrips_direction_and_per_side_proba(tmp_path):
    store = SignalStore(tmp_path / "berich.duckdb")
    short_sig = Signal(
        date=pd.Timestamp("2024-01-05"),
        ticker="TSLA",
        signal=SHORT,
        proba=0.61,
        entry=100.0,
        stop_loss=105.0,  # short stop sits ABOVE entry
        take_profit=90.0,  # short target sits BELOW entry
        size_shares=20,
        notional=2000.0,
        direction="short",
        proba_long=0.30,
        proba_short=0.61,
    )
    store.save([short_sig])
    row = store.history("TSLA").iloc[0]
    assert row["direction"] == "short"
    assert abs(row["proba_short"] - 0.61) < 1e-9
    assert abs(row["proba_long"] - 0.30) < 1e-9
    # Mirrored barriers: target below entry below stop.
    assert row["take_profit"] < row["entry"] < row["stop_loss"]


def test_legacy_signal_defaults_long_direction(tmp_path):
    # A Signal built the old way (no direction kwarg) persists as a long row.
    store = SignalStore(tmp_path / "berich.duckdb")
    store.save([_signal("AAPL", 0.6)])
    row = store.history("AAPL").iloc[0]
    assert row["direction"] == "long"
