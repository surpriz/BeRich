"""Binary gates that decide whether to take a trade at all.

Gating runs **before** sizing in the risk pipeline: if a gate returns
``False`` for a given event/day, the strategy simply doesn't trade —
position size is 0. The gates are intentionally simple, with explicitly
named parameters so the risk overlay's behavior is fully auditable from
its config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_gate_spy(
    spy_rvol_series: pd.Series,
    *,
    as_of: pd.Timestamp,
    quartile_threshold: float = 0.75,
    lookback_days: int = 252,
) -> bool:
    """Allow the trade iff SPY's recent rvol_20 is below the rolling Q-quartile.

    The threshold is computed on the trailing ``lookback_days`` ending at
    ``as_of`` — strictly causal, no peeking into the test window. Returns
    ``True`` when we *can* trade (i.e. rvol is below the cut-off), ``False``
    when we should sit out.

    A degenerate input (no rvol prior to ``as_of`` or rvol NaN there) is
    treated as "no information → allow the trade".
    """
    if spy_rvol_series is None or spy_rvol_series.empty:
        return True
    prior = spy_rvol_series.loc[spy_rvol_series.index < pd.Timestamp(as_of)]
    if prior.empty:
        return True
    window = prior.tail(lookback_days).dropna()
    if window.empty:
        return True
    current = spy_rvol_series.loc[spy_rvol_series.index <= pd.Timestamp(as_of)].dropna()
    if current.empty:
        return True
    current_value = float(current.iloc[-1])
    if np.isnan(current_value):
        return True
    cutoff = float(window.quantile(quartile_threshold))
    return current_value < cutoff


def drawdown_gate(
    equity_curve: pd.Series,
    *,
    max_dd_threshold: float = 0.20,
) -> bool:
    """Allow new trades iff the current equity is above ``(1 - threshold) * peak``.

    The peak is the running max of ``equity_curve`` to date. The threshold is
    expressed as a positive fraction (0.20 = stop opening new trades after a
    20 % drawdown from peak). Returns ``True`` when we can trade.

    Empty / all-NaN equity_curve returns ``True`` — no history means no
    drawdown signal yet, so don't block the first trades by accident.
    """
    if equity_curve is None or equity_curve.empty:
        return True
    series = equity_curve.dropna()
    if series.empty:
        return True
    peak = float(series.cummax().iloc[-1])
    current = float(series.iloc[-1])
    if peak <= 0:
        return True
    drawdown = 1.0 - current / peak
    return drawdown < max_dd_threshold
