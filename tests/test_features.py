"""Tests for indicators and the feature matrix, including the no-lookahead property."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features import indicators as ind
from berich.features.build import (
    FEATURE_COLUMNS,
    HYG_TICKER,
    LQD_TICKER,
    SECTOR_MAP,
    TLT_TICKER,
    VIX9D_TICKER,
    VIX_TICKER,
    build_features,
)


def _ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    vol = rng.integers(1e3, 1e4, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_rsi_in_bounds():
    df = _ohlcv()
    r = ind.rsi(df["close"], 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_dist_to_rolling_high_is_non_positive():
    df = _ohlcv()
    d = ind.dist_to_rolling_high(df["close"], df["high"], 60).dropna()
    assert (d <= 1e-12).all()


def test_dist_to_rolling_low_is_non_negative():
    df = _ohlcv()
    d = ind.dist_to_rolling_low(df["close"], df["low"], 60).dropna()
    assert (d >= -1e-12).all()


def test_build_features_columns_and_index():
    df = _ohlcv()
    market = _ohlcv(seed=1)
    feats = build_features(df, market=market)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats.index.equals(df.index)


def test_features_have_no_lookahead():
    """Editing a future bar must not change any feature value at earlier bars."""
    df = _ohlcv()
    market = _ohlcv(seed=1)
    base = build_features(df, market=market)

    cut = 200
    perturbed = df.astype(float)
    perturbed.iloc[cut:] *= 1.5  # mangle everything from `cut` onward

    after = build_features(perturbed, market=market)
    # Features strictly before `cut` must be identical (NaNs compared as equal).
    pd.testing.assert_frame_equal(base.iloc[:cut], after.iloc[:cut], check_exact=False, rtol=1e-9)


def test_market_features_have_one_bar_lag():
    """Mangling SPY at bar t must NOT change the cross-asset features for the ticker at t.

    The market frame is shifted by one bar in :func:`build_features`, so a perturbation
    introduced at index ``t`` in SPY can only affect the ticker's row at index ``t+1``
    or later. This is what protects against same-day cross-asset leakage.
    """
    df = _ohlcv()
    market = _ohlcv(seed=1)
    base = build_features(df, market=market)

    cut = 200
    perturbed_market = market.astype(float).copy()
    perturbed_market.iloc[cut:] *= 1.5

    after = build_features(df, market=perturbed_market)
    pd.testing.assert_frame_equal(
        base.iloc[: cut + 1], after.iloc[: cut + 1], check_exact=False, rtol=1e-9
    )


def test_market_features_propagate_after_lag():
    """Sanity check: a SPY perturbation does eventually show up in later ticker rows.

    Pairs with :func:`test_market_features_have_one_bar_lag` to make sure the lag test
    isn't passing by accident — i.e. the market columns are actually wired in.
    """
    df = _ohlcv()
    market = _ohlcv(seed=1)
    base = build_features(df, market=market)

    cut = 200
    perturbed_market = market.astype(float).copy()
    perturbed_market.iloc[cut:] *= 1.5

    after = build_features(df, market=perturbed_market)
    diff = (base.iloc[cut + 1 :] - after.iloc[cut + 1 :]).abs().sum().sum()
    assert diff > 0


def test_calendar_features_match_index_dates():
    df = _ohlcv()
    feats = build_features(df, market=_ohlcv(seed=1))
    idx = pd.DatetimeIndex(df.index)
    expected_month_sin = np.sin(2.0 * np.pi * (idx.month.to_numpy() - 1) / 12.0)
    np.testing.assert_allclose(feats["month_sin"].to_numpy(), expected_month_sin)
    # Business days remaining in the month is always >= 0.
    assert (feats["days_to_month_end"].dropna() >= 0).all()


def _context(seed_offset: int = 100) -> dict[str, pd.DataFrame]:
    """Build a synthetic cross-asset context dict covering every cross-asset family."""
    sector_etfs = sorted(set(SECTOR_MAP.values()))
    tickers = [
        VIX_TICKER,
        VIX9D_TICKER,
        TLT_TICKER,
        HYG_TICKER,
        LQD_TICKER,
        *sector_etfs,
    ]
    return {t: _ohlcv(seed=seed_offset + i) for i, t in enumerate(tickers)}


# Cross-asset columns that derive from `context` (not from the ticker's own OHLCV).
_CONTEXT_FEATURES = [
    "vix_level",
    "vix_ret_5",
    "vix_term",
    "tlt_ret_20",
    "tlt_dist_high_60",
    "credit_spread",
    "sector_rel_ret_20",
]


def test_cross_asset_features_have_one_bar_lag():
    """Perturbing any single context series at bar t must NOT change the ticker's
    cross-asset features at bars ``<= t``. This is the one-bar-lag guarantee that
    protects against same-day leakage from VIX / TLT / credit / sector ETFs.
    """
    df = _ohlcv()
    market = _ohlcv(seed=1)
    context = _context()
    base = build_features(df, market=market, context=context, ticker="AAPL")

    cut = 200
    for source in (VIX_TICKER, VIX9D_TICKER, TLT_TICKER, HYG_TICKER, LQD_TICKER, "XLK"):
        perturbed = {k: v.copy() for k, v in context.items()}
        perturbed[source] = perturbed[source].astype(float).copy()
        perturbed[source].iloc[cut:] *= 1.5
        after = build_features(df, market=market, context=perturbed, ticker="AAPL")
        pd.testing.assert_frame_equal(
            base.iloc[: cut + 1],
            after.iloc[: cut + 1],
            check_exact=False,
            rtol=1e-9,
        )


def test_cross_asset_features_propagate_after_lag():
    """Sanity counter-test: a VIX perturbation does change later rows; the lag test
    isn't passing by accident.
    """
    df = _ohlcv()
    market = _ohlcv(seed=1)
    context = _context()
    base = build_features(df, market=market, context=context, ticker="AAPL")

    cut = 200
    perturbed = {k: v.copy() for k, v in context.items()}
    perturbed[VIX_TICKER] = perturbed[VIX_TICKER].astype(float).copy()
    perturbed[VIX_TICKER].iloc[cut:] *= 1.5
    after = build_features(df, market=market, context=perturbed, ticker="AAPL")

    diff = (
        (base.iloc[cut + 1 :] - after.iloc[cut + 1 :])
        .loc[:, ["vix_level", "vix_ret_5", "vix_term"]]
        .abs()
        .sum()
        .sum()
    )
    assert diff > 0


def test_cross_asset_features_missing_context_yields_nan():
    """When no context is supplied, every cross-asset column except sector_rel
    (which has a neutral 0.0 fallback) must be NaN — i.e. the feature builder
    never silently invents data when sources are absent.
    """
    df = _ohlcv()
    feats = build_features(df, market=_ohlcv(seed=1), context=None, ticker="AAPL")
    nan_cols = [c for c in _CONTEXT_FEATURES if c != "sector_rel_ret_20"]
    for col in nan_cols:
        assert feats[col].isna().all(), f"{col} should be NaN without context"


def test_sector_rel_zero_for_spy_without_mapping():
    """SPY has no sector mapping; the sector_rel_ret_20 feature must fall back to
    0.0 so SPY rows survive the dropna step instead of being silently removed.
    """
    df = _ohlcv()
    feats = build_features(df, market=_ohlcv(seed=1), context=_context(), ticker="SPY")
    assert (feats["sector_rel_ret_20"] == 0.0).all()
