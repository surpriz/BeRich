"""Tests for backtest metrics and the event-based engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.backtest.engine import (
    BacktestConfig,
    _resolve_exit,
    _resolve_exit_trailing,
    _simulate_ticker,
)
from berich.backtest.metrics import compute_metrics, max_drawdown


def test_max_drawdown_known_curve():
    equity = pd.Series([1.0, 1.2, 0.9, 1.5])
    # Peak 1.2 -> trough 0.9 == -25%.
    assert abs(max_drawdown(equity) - (-0.25)) < 1e-9


def test_compute_metrics_positive_drift():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 504))  # ~2 years, positive drift
    m = compute_metrics(rets, trade_returns=[0.05, -0.02, 0.03])
    assert m.total_return > 0
    assert m.sharpe > 0
    assert m.n_trades == 3
    assert abs(m.win_rate - 2 / 3) < 1e-9


def _ramp(n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(100, 130, n)  # steady uptrend -> target should hit
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close}, index=idx
    )


def test_resolve_exit_hits_target():
    df = _ramp()
    _idx, price, reason = _resolve_exit(
        df["high"], df["low"], df["close"], start=1, time_exit=20, stop=90.0, target=110.0
    )
    assert reason == "target"
    assert price == 110.0


def test_simulate_ticker_takes_winning_trade():
    df = _ramp()
    proba = pd.Series(0.9, index=df.index)  # always above threshold
    cfg = BacktestConfig(entry_threshold=0.5, horizon_days=10, atr_window=3)
    daily, trades = _simulate_ticker("TEST", df, proba, cfg, fee=0.0, slip=0.0)
    assert len(trades) >= 1
    assert trades[0].gross_return > 0  # uptrend long is profitable
    assert len(daily) == len(df)
    assert all(t.direction == "long" for t in trades)


def _slide(n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = np.linspace(130, 100, n)  # steady downtrend -> short target should hit
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close}, index=idx
    )


def _equity(daily: pd.Series) -> float:
    return float((1.0 + daily).prod())


def test_resolve_exit_short_stop_hits_on_high():
    high = pd.Series([10.0, 10.0, 13.0, 10.0, 10.0])
    low = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
    close = pd.Series([10.0, 10.0, 12.0, 10.0, 10.0])
    idx, price, reason = _resolve_exit(
        high, low, close, start=1, time_exit=4, stop=12.0, target=8.0, direction="short"
    )
    assert reason == "stop"
    assert idx == 2
    assert price == 12.0


def test_resolve_exit_short_target_hits_on_low():
    high = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
    low = pd.Series([10.0, 10.0, 7.0, 10.0, 10.0])
    close = pd.Series([10.0, 10.0, 8.0, 10.0, 10.0])
    idx, price, reason = _resolve_exit(
        high, low, close, start=1, time_exit=4, stop=12.0, target=8.0, direction="short"
    )
    assert reason == "target"
    assert idx == 2
    assert price == 8.0


def test_simulate_ticker_short_profits_on_downtrend():
    df = _slide()
    proba = pd.Series(0.9, index=df.index)
    cfg = BacktestConfig(entry_threshold=0.5, horizon_days=10, atr_window=3, direction="short")
    daily, trades = _simulate_ticker("TEST", df, proba, cfg, fee=0.0, slip=0.0)
    assert len(trades) >= 1
    assert trades[0].direction == "short"
    assert trades[0].gross_return > 0  # downtrend short is profitable
    assert _equity(daily) > 1.0


def test_borrow_cost_reduces_short_return():
    df = _slide()
    proba = pd.Series(0.9, index=df.index)
    base = BacktestConfig(entry_threshold=0.5, horizon_days=10, atr_window=3, direction="short")
    borrow = base.model_copy(update={"borrow_bps_annual": 1000.0})
    daily_base, _ = _simulate_ticker("TEST", df, proba, base, fee=0.0, slip=0.0)
    daily_borrow, _ = _simulate_ticker("TEST", df, proba, borrow, fee=0.0, slip=0.0)
    assert _equity(daily_borrow) < _equity(daily_base)


def test_long_path_unchanged_vs_explicit_long_direction():
    df = _ramp()
    proba = pd.Series(0.9, index=df.index)
    default_cfg = BacktestConfig(entry_threshold=0.5, horizon_days=10, atr_window=3)
    long_cfg = default_cfg.model_copy(update={"direction": "long"})
    daily_default, _ = _simulate_ticker("TEST", df, proba, default_cfg, fee=0.0, slip=0.0)
    daily_long, _ = _simulate_ticker("TEST", df, proba, long_cfg, fee=0.0, slip=0.0)
    assert daily_default.equals(daily_long)


# ----------------------------------------------------------------- trailing exit ----


def test_resolve_exit_trailing_ratchets_and_exits_on_reversal():
    high = pd.Series([100.0, 103.0, 108.0, 106.0, 101.0])
    low = pd.Series([100.0, 101.0, 106.0, 104.0, 99.0])
    close = pd.Series([100.0, 102.0, 107.0, 105.0, 100.0])
    idx, price, reason = _resolve_exit_trailing(
        high,
        low,
        close,
        start=1,
        time_exit=4,
        entry=100.0,
        init_stop=96.0,
        target=None,
        trail_dist=5.0,
        activation_level=102.0,
    )
    assert reason == "trailing"  # armed, so the ratcheted stop, not the initial fixed one
    assert idx == 4
    assert price == 103.0  # running high 108 - trail 5


def test_resolve_exit_trailing_tp_cap_hits_target_first():
    high = pd.Series([100.0, 105.0, 111.0, 100.0])
    low = pd.Series([100.0, 103.0, 109.0, 100.0])
    close = pd.Series([100.0, 104.0, 110.0, 100.0])
    idx, price, reason = _resolve_exit_trailing(
        high,
        low,
        close,
        start=1,
        time_exit=3,
        entry=100.0,
        init_stop=96.0,
        target=110.0,
        trail_dist=5.0,
        activation_level=102.0,
    )
    assert reason == "target"
    assert idx == 2
    assert price == 110.0


def test_resolve_exit_trailing_is_causal_no_same_bar_lookahead():
    # Bar 1 spikes (high 110) but fully retraces (low 99) within the same bar — it must not
    # self-trigger. The ratcheted stop (~105) only bites on bar 2.
    high = pd.Series([100.0, 110.0, 106.0, 105.0])
    low = pd.Series([100.0, 99.0, 104.0, 103.0])
    close = pd.Series([100.0, 100.0, 104.0, 104.0])
    idx, price, reason = _resolve_exit_trailing(
        high,
        low,
        close,
        start=1,
        time_exit=3,
        entry=100.0,
        init_stop=96.0,
        target=None,
        trail_dist=5.0,
        activation_level=102.0,
    )
    assert idx == 2  # not 1 — the spike bar did not set-and-trigger its own stop
    assert reason == "trailing"
    assert price == 105.0


def test_resolve_exit_trailing_pre_activation_stop_is_plain_stop():
    high = pd.Series([100.0, 100.0])
    low = pd.Series([100.0, 95.0])  # breaks the initial stop before the trail ever arms
    close = pd.Series([100.0, 96.0])
    _idx, price, reason = _resolve_exit_trailing(
        high,
        low,
        close,
        start=1,
        time_exit=1,
        entry=100.0,
        init_stop=96.0,
        target=None,
        trail_dist=5.0,
        activation_level=102.0,
    )
    assert reason == "stop"
    assert price == 96.0


def test_simulate_ticker_trailing_captures_more_than_fixed_on_uptrend():
    df = _ramp()
    proba = pd.Series(0.9, index=df.index)
    fixed = BacktestConfig(entry_threshold=0.5, horizon_days=10, atr_window=3)
    trailing = fixed.model_copy(update={"exit_mode": "trailing"})
    _, trades_f = _simulate_ticker("TEST", df, proba, fixed, fee=0.0, slip=0.0)
    _, trades_t = _simulate_ticker("TEST", df, proba, trailing, fee=0.0, slip=0.0)
    assert trades_t[0].gross_return > trades_f[0].gross_return  # let the winner run
    assert trades_t[0].reason in {"trailing", "time"}
