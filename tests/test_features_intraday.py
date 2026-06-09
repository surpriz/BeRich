"""Intraday feature set: calendar swap, gap_open drop, and the no-lookahead property."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features.build import (
    DAILY_CALENDAR_COLUMNS,
    INTRADAY_CALENDAR_COLUMNS,
    INTRADAY_FEATURE_COLUMNS,
    build_features,
    feature_columns,
)


def _hourly_ohlcv(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    vol = rng.integers(1e3, 1e4, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_intraday_columns_swap_calendar_for_cyclicals():
    df = _hourly_ohlcv()
    market = _hourly_ohlcv(seed=1)
    feats = build_features(df, market=market, intraday=True)

    assert list(feats.columns) == INTRADAY_FEATURE_COLUMNS
    # Daily calendar features are gone; intraday cyclicals are present.
    for col in DAILY_CALENDAR_COLUMNS:
        assert col not in feats.columns
    for col in INTRADAY_CALENDAR_COLUMNS:
        assert col in feats.columns


def test_intraday_cyclicals_in_unit_range():
    df = _hourly_ohlcv()
    feats = build_features(df, intraday=True)
    for col in INTRADAY_CALENDAR_COLUMNS:
        vals = feats[col].dropna()
        assert (vals >= -1.0 - 1e-12).all() and (vals <= 1.0 + 1e-12).all()


def test_intraday_micro_drops_gap_open():
    cols = feature_columns(micro=True, intraday=True)
    assert "gap_open" not in cols
    # Daily micro still carries gap_open.
    assert "gap_open" in feature_columns(micro=True, intraday=False)


def test_intraday_features_have_no_lookahead():
    df = _hourly_ohlcv()
    market = _hourly_ohlcv(seed=1)
    base = build_features(df, market=market, intraday=True)

    cut = 250
    perturbed = df.astype(float)
    perturbed.iloc[cut:] *= 1.5

    after = build_features(perturbed, market=market, intraday=True)
    pd.testing.assert_frame_equal(base.iloc[:cut], after.iloc[:cut], check_exact=False, rtol=1e-9)
