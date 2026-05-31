"""Microstructure / liquidity features derived from daily OHLCV.

A genuinely new feature *family* — liquidity, effective spread, and intraday position —
distinct from the momentum / trend / volatility / macro space the prior phases exhausted.
These are short-horizon-relevant (liquidity and bid-ask bounce drive day-to-day reversal
and continuation) and computed purely from the OHLCV already on disk, so they add no fragile
data dependency. All are causal: each value uses only data up to and including bar ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MICRO_FEATURE_COLUMNS: list[str] = [
    "clv",  # close location value: where in the day's range it closed (-1..+1)
    "gap_open",  # overnight gap: open / prev_close - 1
    "hl_range",  # daily high-low range as a fraction of close (intraday vol proxy)
    "parkinson_10",  # 10-day Parkinson volatility from the high-low range
    "amihud_20",  # 20-day Amihud illiquidity (|return| per dollar volume), log-scaled
    "roll_spread_20",  # Roll effective-spread estimator from price-change autocovariance
]

_EPS = 1e-12


def build_micro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the microstructure feature matrix aligned to ``df`` (causal; warm-ups NaN)."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)

    feats = pd.DataFrame(index=df.index)

    rng = (high - low).replace(0.0, np.nan)
    feats["clv"] = (((close - low) - (high - close)) / rng).fillna(0.0)
    feats["gap_open"] = open_ / close.shift(1) - 1.0
    feats["hl_range"] = (high - low) / close

    # Parkinson volatility: sqrt( mean( ln(H/L)^2 ) / (4 ln2) ) over a 10-day window.
    log_hl_sq = np.log((high / low).replace([np.inf, -np.inf], np.nan)) ** 2
    feats["parkinson_10"] = np.sqrt(
        log_hl_sq.rolling(10, min_periods=10).mean() / (4.0 * np.log(2.0))
    )

    # Amihud illiquidity: average |daily return| per dollar of volume, over 20 days.
    ret_1 = close.pct_change(fill_method=None)
    dollar_vol = (close * volume).replace(0.0, np.nan)
    illiq = (ret_1.abs() / dollar_vol).rolling(20, min_periods=20).mean()
    feats["amihud_20"] = np.log1p(illiq * 1e9)  # log-scale the tiny per-dollar number

    # Roll effective spread: 2*sqrt(-cov(Δp_t, Δp_{t-1})) when the autocovariance is negative
    # (the bid-ask bounce signature); 0 otherwise. Expressed as a fraction of price.
    dprice = close.diff()
    cov = dprice.rolling(20, min_periods=20).cov(dprice.shift(1))
    feats["roll_spread_20"] = 2.0 * np.sqrt((-cov).clip(lower=0.0)) / (close + _EPS)

    return feats


__all__ = ["MICRO_FEATURE_COLUMNS", "build_micro_features"]
