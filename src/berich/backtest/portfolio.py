"""Core-satellite portfolio engine (Phase 9).

Takes a dict of named strategies (each a daily-returns ``pd.Series``) and a
weighting policy, returns a daily portfolio-return series + the usual
performance metrics. Two policies are supported:

- **Static weights**: the same weight vector every day. Useful for the
  grid sweep ("80/20/0").
- **Walk-forward weights**: a sequence of ``(start_date, weights)``
  tuples produced by an optimizer on training folds. Applied chronologically
  so the test fold never sees its own weights.

Rebalancing: the portfolio is rebalanced to its target weights on a
calendar schedule (default monthly, first business day). Between rebalances
weights drift with realized returns — same convention as a real fund. Each
rebalance pays ``cost_bps`` x turnover, where turnover is the L1 distance
between pre- and post-rebalance weights. With static weights and small
drifts this cost is tiny; with walk-forward swings it can be material.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from berich.backtest.metrics import PerfMetrics, compute_metrics

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_COST_BPS = 1.5
DEFAULT_REBALANCE = "M"  # monthly


@dataclass
class PortfolioBacktestResult:
    """Daily portfolio returns + summary metrics + per-rebalance turnover."""

    metrics: PerfMetrics
    returns: pd.Series  # daily portfolio returns net of rebalancing cost
    cumulative: pd.Series  # equity curve starting at 1.0
    turnover: pd.Series  # turnover paid at each rebalance day


def _stack_strategies(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    """Align named strategy returns onto a shared daily index, NaN → 0."""
    if not strategies:
        msg = "no strategies supplied"
        raise ValueError(msg)
    frame = pd.DataFrame(strategies).sort_index()
    return frame.fillna(0.0)


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """First business-day of each ``freq`` period present in ``index``.

    ``freq`` supports the pandas offset codes: ``"M"`` (monthly), ``"Q"``,
    ``"Y"``. We snap to the first index date that falls in each period
    (rather than the period's calendar boundary) so an empty period
    contributes no rebalance.
    """
    if freq == "M":
        period = index.to_period("M")
    elif freq == "Q":
        period = index.to_period("Q")
    elif freq == "Y":
        period = index.to_period("Y")
    else:
        msg = f"unsupported rebalance frequency: {freq!r}"
        raise ValueError(msg)
    df = pd.DataFrame({"period": period, "date": index})
    first = df.groupby("period", observed=True)["date"].min()
    return pd.DatetimeIndex(first.values)


def run_portfolio_backtest(
    strategies: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    *,
    walk_forward_weights: Iterable[tuple[pd.Timestamp, dict[str, float]]] | None = None,
    rebalance: Literal["M", "Q", "Y"] = DEFAULT_REBALANCE,
    cost_bps: float = DEFAULT_COST_BPS,
) -> PortfolioBacktestResult:
    """Run a portfolio backtest with either static or walk-forward weights.

    Exactly one of ``weights`` and ``walk_forward_weights`` must be supplied.
    Returns the daily NET portfolio returns (rebalancing cost subtracted on
    the rebalance day) and the running equity curve.
    """
    if (weights is None) == (walk_forward_weights is None):
        msg = "provide exactly one of `weights` or `walk_forward_weights`"
        raise ValueError(msg)

    frame = _stack_strategies(strategies)
    names = list(frame.columns)
    cost = cost_bps / 1e4

    # Build per-day target-weight series. For static, the same row everywhere.
    # For walk-forward, propagate each (start_date, weights) until the next.
    if weights is not None:
        target = pd.DataFrame(
            np.tile([weights.get(name, 0.0) for name in names], (len(frame), 1)),
            index=frame.index,
            columns=pd.Index(names),
        )
    else:
        assert walk_forward_weights is not None  # narrowed for ty  # noqa: S101
        target = pd.DataFrame(np.nan, index=frame.index, columns=pd.Index(names))
        for start_date, w in walk_forward_weights:
            row = np.array([w.get(name, 0.0) for name in names])
            target.loc[target.index >= pd.Timestamp(start_date)] = row
        target = target.ffill().fillna(0.0)

    rebal_days = set(_rebalance_dates(pd.DatetimeIndex(frame.index), rebalance))
    rebal_days.add(frame.index[0])  # first day is always a rebalance to seed weights

    # Drifted weights = previous-day weights * (1 + r) then renormalized.
    actual = pd.DataFrame(0.0, index=frame.index, columns=pd.Index(names))
    portfolio_returns = pd.Series(0.0, index=frame.index)
    turnover_series = pd.Series(0.0, index=frame.index)

    current = target.iloc[0].to_numpy().copy()
    actual.iloc[0] = current

    for i in range(1, len(frame)):
        # Drift the previous-day weights with that day's return…
        returns_today = frame.iloc[i].to_numpy()
        drifted = current * (1.0 + returns_today)
        portfolio_returns.iloc[i] = float(np.sum(current * returns_today))
        if drifted.sum() <= 0:
            drifted = current.copy()
        drifted = drifted / drifted.sum()

        if frame.index[i] in rebal_days:
            target_today = target.iloc[i].to_numpy()
            turnover = float(np.abs(drifted - target_today).sum())
            portfolio_returns.iloc[i] -= cost * turnover
            turnover_series.iloc[i] = turnover
            current = target_today
        else:
            current = drifted
        actual.iloc[i] = current

    cumulative = (1.0 + portfolio_returns).cumprod()
    metrics = compute_metrics(portfolio_returns)
    return PortfolioBacktestResult(
        metrics=metrics,
        returns=portfolio_returns,
        cumulative=cumulative,
        turnover=turnover_series,
    )
