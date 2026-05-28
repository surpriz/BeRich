"""Tests for indicators and the feature matrix, including the no-lookahead property."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features import indicators as ind
from berich.features.build import FEATURE_COLUMNS, build_features


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
