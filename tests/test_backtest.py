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
