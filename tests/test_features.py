"""Tests for indicators and the feature matrix, including the no-lookahead property."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features import indicators as ind
from berich.features.build import FEATURE_COLUMNS, build_features


def _ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
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


def test_build_features_columns_and_index():
    df = _ohlcv()
    feats = build_features(df)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats.index.equals(df.index)


def test_features_have_no_lookahead():
    """Editing a future bar must not change any feature value at earlier bars."""
    df = _ohlcv()
    base = build_features(df)

    cut = 150
    perturbed = df.astype(float)
    perturbed.iloc[cut:] *= 1.5  # mangle everything from `cut` onward

    after = build_features(perturbed)
    # Features strictly before `cut` must be identical (NaNs compared as equal).
    pd.testing.assert_frame_equal(base.iloc[:cut], after.iloc[:cut], check_exact=False, rtol=1e-9)
