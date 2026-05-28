"""Assemble the model feature matrix from an OHLCV frame.

`build_features` returns a frame aligned to the input index with one column per
feature. All features are causal (see :mod:`berich.features.indicators`). The
``FEATURE_COLUMNS`` constant is the canonical, ordered feature list used by every
model so training and inference never disagree on column order.

Cross-asset market-regime features (SPY trailing return / vol) are optional and
broadcast via the ``market`` argument. They are shifted by one bar before being
joined to the ticker frame so a same-day SPY observation can never leak into the
features for any other ticker at the same date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features import indicators as ind

# Ticker used as the broad-market regime proxy in the cross-asset features.
MARKET_TICKER = "SPY"

# Lookback window for cross-asset SPY regime features.
MARKET_LOOKBACK = 20

# Lookback window for distance-to-rolling-high/low mean-reversion features.
DIST_LOOKBACK = 60

FEATURE_COLUMNS: list[str] = [
    "ret_1",
    "ret_5",
    "mom_10",
    "mom_20",
    "mom_60",
    "mom_120",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_pct",
    "rvol_20",
    "close_sma20_ratio",
    "close_sma50_ratio",
    "volume_z20",
    "dist_high_60",
    "dist_low_60",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "days_to_month_end",
    "spy_ret_20",
    "spy_rvol_20",
]


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sinusoidal day-of-week / month encodings plus business days to month-end.

    Sinusoidal encoding preserves cyclical proximity (Friday is close to Monday,
    December to January) so tree and gradient models don't have to discover the
    wrap-around on their own. ``days_to_month_end`` counts the *business* days
    remaining in the calendar month, end-of-month effects being a well-known
    seasonality in equities.
    """
    out = pd.DataFrame(index=index)
    # 0=Mon … 6=Sun (data is business days, so values land in 0..4 in practice).
    index_series = pd.Series(index)
    dow = index_series.dt.dayofweek.to_numpy()
    month = index_series.dt.month.to_numpy()
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 5.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 5.0)
    out["month_sin"] = np.sin(2.0 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * (month - 1) / 12.0)

    # ``BMonthEnd(0)`` rolls to the same date if it is already a business month-end,
    # otherwise to the next one — always within the same month, no future lookahead.
    month_end = index + pd.offsets.BMonthEnd(0)
    days = np.busday_count(
        index.to_numpy().astype("datetime64[D]"),
        month_end.to_numpy().astype("datetime64[D]"),
    )
    out["days_to_month_end"] = days.astype(float)
    return out


def _market_regime_features(
    target_index: pd.DatetimeIndex,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """SPY trailing return / realized vol, lagged one bar and reindexed to ``target_index``.

    The lag is what protects against cross-asset same-bar leakage: feature at row
    ``t`` only sees market data through ``t-1``. Missing market dates produce NaN,
    which the caller drops along with other warm-up rows.
    """
    close = market["close"]
    spy_ret = close.pct_change(MARKET_LOOKBACK)
    log_ret = np.log(close / close.shift(1))
    spy_rvol = log_ret.rolling(MARKET_LOOKBACK, min_periods=MARKET_LOOKBACK).std()

    feats = pd.DataFrame({"spy_ret_20": spy_ret, "spy_rvol_20": spy_rvol})
    feats = feats.shift(1)  # one-bar lag — no same-day leakage into other tickers
    return feats.reindex(target_index)


def build_features(
    df: pd.DataFrame,
    *,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the canonical feature matrix from an OHLCV frame.

    Returns a frame with exactly ``FEATURE_COLUMNS``, indexed like ``df``. Rows where
    any feature is still warming up are left as NaN for the caller to drop. When
    ``market`` is omitted the SPY regime columns are NaN, which the join+dropna
    pipeline filters out before training.
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    feats = pd.DataFrame(index=df.index)

    feats["ret_1"] = close.pct_change(1)
    feats["ret_5"] = close.pct_change(5)
    feats["mom_10"] = ind.momentum(close, 10)
    feats["mom_20"] = ind.momentum(close, 20)
    feats["mom_60"] = ind.momentum(close, 60)
    feats["mom_120"] = ind.momentum(close, 120)
    feats["rsi_14"] = ind.rsi(close, 14)

    # Normalize MACD components by price so they are stationary across price levels
    # and time (raw MACD scales with the absolute price and drifts heavily otherwise).
    macd = ind.macd(close).div(close, axis=0)
    feats[["macd", "macd_signal", "macd_hist"]] = macd

    # ATR as a fraction of price — comparable across tickers and price levels.
    feats["atr_pct"] = ind.atr(high, low, close, 14) / close
    feats["rvol_20"] = ind.realized_vol(close, 20)

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    feats["close_sma20_ratio"] = close / sma20 - 1.0
    feats["close_sma50_ratio"] = close / sma50 - 1.0

    feats["volume_z20"] = ind.zscore(volume.astype(float), 20)

    feats["dist_high_60"] = ind.dist_to_rolling_high(close, high, DIST_LOOKBACK)
    feats["dist_low_60"] = ind.dist_to_rolling_low(close, low, DIST_LOOKBACK)

    calendar = _calendar_features(pd.DatetimeIndex(df.index))
    for col in ("dow_sin", "dow_cos", "month_sin", "month_cos", "days_to_month_end"):
        feats[col] = calendar[col]

    if market is not None:
        regime = _market_regime_features(pd.DatetimeIndex(df.index), market)
        feats["spy_ret_20"] = regime["spy_ret_20"]
        feats["spy_rvol_20"] = regime["spy_rvol_20"]
    else:
        feats["spy_ret_20"] = np.nan
        feats["spy_rvol_20"] = np.nan

    return feats[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
