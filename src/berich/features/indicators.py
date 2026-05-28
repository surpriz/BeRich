"""Causal technical indicators.

Every function uses only past and current bars (no lookahead): each value at time
``t`` depends solely on data up to and including ``t``. Implemented with pandas/numpy
to avoid the numpy-2 incompatibility in pandas-ta. Leading values are NaN until the
window fills; callers drop those rows before training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When there are no losses RSI is 100 by definition.
    out = out.where(avg_loss != 0.0, 100.0)
    out.name = f"rsi_{window}"
    return out


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": sig, "macd_hist": line - sig},
        index=close.index,
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: max of the three classic ranges, using the prior close."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    tr = true_range(high, low, close)
    out = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    out.name = f"atr_{window}"
    return out


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    """Rate of change over ``window`` bars (fractional)."""
    out = close.pct_change(window)
    out.name = f"mom_{window}"
    return out


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of daily log returns (realized volatility)."""
    log_ret = np.log(close / close.shift(1))
    out = log_ret.rolling(window, min_periods=window).std()
    out.name = f"rvol_{window}"
    return out


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score of a series against its own trailing window."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def dist_to_rolling_high(close: pd.Series, high: pd.Series, window: int) -> pd.Series:
    """Fractional gap between close and the trailing ``window``-bar rolling high.

    Value is in (-1, 0]: 0 when today's close is itself the rolling high; negative
    otherwise (a mean-reversion proxy for how far we've fallen from the recent peak).
    """
    roll_high = high.rolling(window, min_periods=window).max()
    out = close / roll_high - 1.0
    out.name = f"dist_high_{window}"
    return out


def dist_to_rolling_low(close: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Fractional gap between close and the trailing ``window``-bar rolling low.

    Value is in [0, +inf): 0 when today's close is itself the rolling low; positive
    otherwise (a mean-reversion proxy for how far we've risen from the recent trough).
    """
    roll_low = low.rolling(window, min_periods=window).min()
    out = close / roll_low - 1.0
    out.name = f"dist_low_{window}"
    return out
