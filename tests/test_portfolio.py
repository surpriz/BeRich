"""Phase 9 portfolio engine tests: weights, rebalance, turnover, strategy series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from berich.backtest.portfolio import (
    _rebalance_dates,
    _stack_strategies,
    run_portfolio_backtest,
)
from berich.backtest.strategies import build_bnh_returns, build_calendar_returns


def _series(values: list[float], start: str = "2020-01-01", name: str = "s") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"), name=name)


def test_stack_strategies_aligns_and_fills_nan():
    s1 = _series([0.01, 0.02], name="a")
    s2 = pd.Series([0.005], index=[s1.index[0]], name="b")
    out = _stack_strategies({"a": s1, "b": s2})
    assert list(out.columns) == ["a", "b"]
    # Day 2 had no "b" value → filled with 0.
    assert out.loc[s1.index[1], "b"] == 0.0


def test_rebalance_dates_monthly_picks_first_business_day_of_each_month():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    days = _rebalance_dates(idx, "M")
    months = {(d.year, d.month) for d in days}
    assert months == {(2020, 1), (2020, 2), (2020, 3)}


# ------------------------------------------------------- portfolio engine ----


def test_static_weight_portfolio_equals_weighted_average():
    """With static 50/50 the daily portfolio return is the mean of the two series."""
    a = _series([0.01, 0.02, 0.0, -0.01], name="a")
    b = _series([0.0, -0.01, 0.02, 0.01], name="b")
    result = run_portfolio_backtest({"a": a, "b": b}, weights={"a": 0.5, "b": 0.5}, cost_bps=0.0)
    expected_day1 = 0.5 * 0.02 + 0.5 * (-0.01)  # day 1 = index 1 (day 0 is seed)
    assert result.returns.iloc[1] == pytest.approx(expected_day1, rel=1e-9)


def test_full_weight_on_one_strategy_matches_that_strategy():
    a = _series([0.01, -0.02, 0.03, 0.0], name="a")
    b = _series([0.10, 0.10, 0.10, 0.10], name="b")
    result = run_portfolio_backtest({"a": a, "b": b}, weights={"a": 1.0, "b": 0.0}, cost_bps=0.0)
    # Excluding the seed bar (day 0) the portfolio return equals series ``a``.
    pd.testing.assert_series_equal(
        result.returns.iloc[1:].reset_index(drop=True),
        a.iloc[1:].reset_index(drop=True),
        check_names=False,
        check_exact=False,
        atol=1e-9,
    )


def test_zero_cost_rebalance_does_not_change_returns_when_weights_static():
    a = _series([0.01] * 30, name="a")
    b = _series([0.01] * 30, name="b")
    result_costly = run_portfolio_backtest(
        {"a": a, "b": b}, weights={"a": 0.5, "b": 0.5}, cost_bps=10.0
    )
    # With identical-returning series, rebalancing drift is zero → turnover zero
    # → cost zero even at 10 bps.
    assert float(result_costly.turnover.sum()) == pytest.approx(0.0, abs=1e-12)


def test_walk_forward_weights_propagate_chronologically():
    """Switch from 100% a to 100% b at the schedule boundary; the average
    portfolio return must be closer to a's return before the switch and
    closer to b's after (allowing for monthly rebalance + drift)."""
    a = _series([0.01] * 40, name="a")
    b = _series([0.02] * 40, name="b")
    half = a.index[20]
    schedule = [
        (a.index[0], {"a": 1.0, "b": 0.0}),
        (half, {"a": 0.0, "b": 1.0}),
    ]
    result = run_portfolio_backtest({"a": a, "b": b}, walk_forward_weights=schedule, cost_bps=0.0)
    pre = result.returns.loc[result.returns.index < half].iloc[1:]
    post = result.returns.loc[result.returns.index >= half]
    # Pre period should be much closer to a's return (0.01) than to b's (0.02).
    assert abs(pre.mean() - 0.01) < abs(pre.mean() - 0.02)
    # Post period (after the rebalance has propagated) should be closer to b's.
    assert abs(post.mean() - 0.02) < abs(post.mean() - 0.01)


def test_run_portfolio_backtest_rejects_both_weight_modes():
    a = _series([0.01], name="a")
    with pytest.raises(ValueError, match="exactly one"):
        run_portfolio_backtest({"a": a})
    with pytest.raises(ValueError, match="exactly one"):
        run_portfolio_backtest(
            {"a": a},
            weights={"a": 1.0},
            walk_forward_weights=[(a.index[0], {"a": 1.0})],
        )


# -------------------------------------------------- strategy return builders ----


class _MockStore:
    """In-memory OhlcvStore stand-in for the unit tests."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = {k.upper(): v for k, v in frames.items()}

    def load(self, ticker: str) -> pd.DataFrame | None:
        return self._frames.get(ticker.upper())


def _ohlcv(values: list[float], start: str = "2020-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(values))
    arr = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr * 1.01, "low": arr * 0.99, "close": arr, "volume": 1e6},
        index=idx,
    )


def test_build_bnh_returns_matches_pct_change():
    closes = [100.0, 101.0, 99.0, 102.0]
    store = _MockStore({"SPY": _ohlcv(closes)})
    ret = build_bnh_returns(store)
    # Day 0 is NaN→0; days 1..3 are simple returns.
    assert ret.iloc[0] == 0.0
    assert ret.iloc[1] == pytest.approx(0.01, rel=1e-9)
    assert ret.iloc[2] == pytest.approx(-0.0198, rel=1e-3)


def test_build_calendar_returns_is_zero_outside_window():
    """Build a single-month frame so the TOM window covers only the first
    and last 3 bdays; days in the middle must return exactly 0 (no exposure).
    """
    closes = list(np.linspace(100, 110, 20))  # 20 bdays starting Jan 2 → all in Jan
    store = _MockStore({"SPY": _ohlcv(closes, start="2020-01-02")})
    ret = build_calendar_returns(store, window=3)
    in_window = (ret != 0).to_numpy()
    # First 3 and last 3 indices should be active.
    assert in_window[:3].any()
    assert in_window[-3:].any()
    # Mid-month (well clear of either window) must be flat.
    mid_start, mid_end = 6, 14
    assert not in_window[mid_start:mid_end].any()
