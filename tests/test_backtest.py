"""Tests for backtest metrics and the event-based engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.backtest.engine import BacktestConfig, _resolve_exit, _simulate_ticker
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
