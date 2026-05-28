"""Assemble the model feature matrix from an OHLCV frame.

`build_features` returns a frame aligned to the input index with one column per
feature. All features are causal (see :mod:`berich.features.indicators`). The
``FEATURE_COLUMNS`` constant is the canonical, ordered feature list used by every
model so training and inference never disagree on column order.

Cross-asset features (SPY regime, VIX, rates, credit, sector-relative) are
optional and broadcast via the ``market`` and ``context`` arguments. Every
cross-asset value is shifted by one bar before being joined to the ticker frame
so a same-day observation on a different symbol can never leak into the features
for this ticker at the same date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features import indicators as ind

# Ticker used as the broad-market regime proxy in the SPY features.
MARKET_TICKER = "SPY"

# Lookback window for cross-asset SPY regime features.
MARKET_LOOKBACK = 20

# Lookback window for distance-to-rolling-high/low mean-reversion features.
DIST_LOOKBACK = 60

# Lookback window for the TLT (rates) and sector-relative trailing returns.
RATES_LOOKBACK = 20

# Window for the short VIX delta — five bars matches a trading week.
VIX_DELTA = 5

# Yfinance symbols used by the cross-asset features.
VIX_TICKER = "^VIX"
VIX9D_TICKER = "^VIX9D"
TLT_TICKER = "TLT"
HYG_TICKER = "HYG"
LQD_TICKER = "LQD"

# Watchlist ticker → sector-ETF ticker for the sector-relative momentum feature.
# Tickers absent from the map (e.g. SPY itself) get a sector_rel value of 0.0 so
# they aren't silently dropped by the join+dropna pipeline.
SECTOR_MAP: dict[str, str] = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "AMZN": "XLY",
    "GOOGL": "XLC",
    "META": "XLC",
    "JPM": "XLF",
    "XOM": "XLE",
    "JNJ": "XLV",
}

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
    "month_sin",
    "month_cos",
    "days_to_month_end",
    "spy_ret_20",
    "spy_rvol_20",
    "vix_level",
    "vix_ret_5",
    "vix_term",
    "tlt_ret_20",
    "tlt_dist_high_60",
    "credit_spread",
    "sector_rel_ret_20",
]


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sinusoidal monthly encoding plus business days to month-end.

    ``days_to_month_end`` counts the *business* days remaining in the calendar
    month, end-of-month effects being a well-known seasonality in equities.
    """
    out = pd.DataFrame(index=index)
    # day-of-week proved to be ~0% LGBM importance, so it's no longer carried; only
    # monthly seasonality (sin/cos) and days-to-month-end survive.
    index_series = pd.Series(index)
    month = index_series.dt.month.to_numpy()
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


def _lagged_reindex(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Shift a cross-asset series by one bar and reindex to the target ticker's dates."""
    shifted = series.shift(1)
    # ty narrows shift+reindex to Series|DataFrame; explicit cast keeps callers tidy.
    return pd.Series(shifted.reindex(target_index))


def _vix_features(
    target_index: pd.DatetimeIndex,
    context: dict[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    """VIX level, 5-day delta, and term-structure ratio (VIX9D / VIX), 1-bar lagged.

    A ``vix_term`` below 1 means the front of the curve is below the medium part —
    typical of calm regimes; above 1 it signals near-term stress, a well-documented
    risk indicator. NaN-tolerant: if either VIX series is absent we emit NaN
    instead of fabricating zeros.
    """
    vix = context.get(VIX_TICKER)
    if vix is None or vix.empty:
        nan = pd.Series(np.nan, index=target_index)
        return {"vix_level": nan, "vix_ret_5": nan, "vix_term": nan}
    vix_close = vix["close"]
    vix_ret = vix_close.pct_change(VIX_DELTA)

    out: dict[str, pd.Series] = {
        "vix_level": _lagged_reindex(vix_close, target_index),
        "vix_ret_5": _lagged_reindex(vix_ret, target_index),
    }

    vix9d = context.get(VIX9D_TICKER)
    if vix9d is not None and not vix9d.empty:
        # Ratio computed on the same date, then lagged together — preserves the
        # term-structure signal at t-1 without mixing dates from different lags.
        ratio = vix9d["close"] / vix_close.replace(0.0, np.nan)
        out["vix_term"] = _lagged_reindex(ratio, target_index)
    else:
        out["vix_term"] = pd.Series(np.nan, index=target_index)
    return out


def _rates_features(
    target_index: pd.DatetimeIndex,
    context: dict[str, pd.DataFrame],
) -> dict[str, pd.Series]:
    """TLT trailing return and distance to rolling 60-bar high, 1-bar lagged."""
    tlt = context.get(TLT_TICKER)
    if tlt is None or tlt.empty:
        nan = pd.Series(np.nan, index=target_index)
        return {"tlt_ret_20": nan, "tlt_dist_high_60": nan}
    close = tlt["close"]
    high = tlt["high"]
    return {
        "tlt_ret_20": _lagged_reindex(close.pct_change(RATES_LOOKBACK), target_index),
        "tlt_dist_high_60": _lagged_reindex(
            ind.dist_to_rolling_high(close, high, DIST_LOOKBACK),
            target_index,
        ),
    }


def _credit_spread_feature(
    target_index: pd.DatetimeIndex,
    context: dict[str, pd.DataFrame],
) -> pd.Series:
    """HYG ret_20 - LQD ret_20: a daily risk-on / risk-off credit proxy, lagged 1 bar."""
    hyg = context.get(HYG_TICKER)
    lqd = context.get(LQD_TICKER)
    if hyg is None or hyg.empty or lqd is None or lqd.empty:
        return pd.Series(np.nan, index=target_index)
    spread = hyg["close"].pct_change(RATES_LOOKBACK) - lqd["close"].pct_change(RATES_LOOKBACK)
    return _lagged_reindex(spread, target_index)


def _sector_relative_return(
    target_index: pd.DatetimeIndex,
    ticker: str | None,
    own_close: pd.Series,
    context: dict[str, pd.DataFrame],
) -> pd.Series:
    """Ticker ret_20 minus its sector ETF's ret_20, lagged 1 bar.

    Captures sector-adjusted momentum: did the name lead or lag its sector over
    the trailing month, *as of the prior close*? Tickers without a sector mapping
    (SPY today) receive 0.0 — neutral so the row survives the dropna step.
    """
    if ticker is None:
        return pd.Series(np.nan, index=target_index)
    sector = SECTOR_MAP.get(ticker.upper())
    if sector is None:
        # Tickers like SPY have no sector context; neutral value keeps them in
        # the panel rather than silently dropping a benchmark constituent.
        return pd.Series(0.0, index=target_index)
    sector_df = context.get(sector)
    if sector_df is None or sector_df.empty:
        return pd.Series(np.nan, index=target_index)
    own_ret = own_close.pct_change(RATES_LOOKBACK)
    sector_ret = sector_df["close"].pct_change(RATES_LOOKBACK)
    diff = own_ret - sector_ret.reindex(own_close.index)
    return _lagged_reindex(diff, target_index)


def build_features(
    df: pd.DataFrame,
    *,
    market: pd.DataFrame | None = None,
    context: dict[str, pd.DataFrame] | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Compute the canonical feature matrix from an OHLCV frame.

    Returns a frame with exactly ``FEATURE_COLUMNS``, indexed like ``df``. Rows where
    any feature is still warming up are left as NaN for the caller to drop. When
    ``market`` or ``context`` is omitted the corresponding cross-asset columns are
    NaN, which the join+dropna pipeline filters out before training. ``ticker`` is
    used to look up the sector ETF for the sector-relative feature.
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
    for col in ("month_sin", "month_cos", "days_to_month_end"):
        feats[col] = calendar[col]

    target_index = pd.DatetimeIndex(df.index)
    if market is not None:
        regime = _market_regime_features(target_index, market)
        feats["spy_ret_20"] = regime["spy_ret_20"]
        feats["spy_rvol_20"] = regime["spy_rvol_20"]
    else:
        feats["spy_ret_20"] = np.nan
        feats["spy_rvol_20"] = np.nan

    ctx = context or {}
    vix = _vix_features(target_index, ctx)
    feats["vix_level"] = vix["vix_level"]
    feats["vix_ret_5"] = vix["vix_ret_5"]
    feats["vix_term"] = vix["vix_term"]
    rates = _rates_features(target_index, ctx)
    feats["tlt_ret_20"] = rates["tlt_ret_20"]
    feats["tlt_dist_high_60"] = rates["tlt_dist_high_60"]
    feats["credit_spread"] = _credit_spread_feature(target_index, ctx)
    feats["sector_rel_ret_20"] = _sector_relative_return(target_index, ticker, close, ctx)

    return feats[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
