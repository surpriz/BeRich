"""Tests for the microstructure feature family (causality, shape, integration)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features.build import build_features, feature_columns
from berich.features.microstructure import MICRO_FEATURE_COLUMNS, build_micro_features


def _ohlcv(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def test_micro_columns_present_and_finite_after_warmup():
    feats = build_micro_features(_ohlcv())
    assert list(feats.columns) == MICRO_FEATURE_COLUMNS
    warm = feats.iloc[30:]  # past the 20-day windows
    assert np.isfinite(warm.to_numpy()).all()


def test_clv_in_range():
    feats = build_micro_features(_ohlcv())
    assert feats["clv"].between(-1.0, 1.0).all()


def test_micro_is_causal():
    """Appending a future bar must not change any earlier microstructure value."""
    df = _ohlcv(120, seed=1)
    full = build_micro_features(df)
    truncated = build_micro_features(df.iloc[:-1])
    common = truncated.index[30:]  # ignore warm-up region
    pd.testing.assert_frame_equal(full.loc[common], truncated.loc[common])


def test_feature_columns_appends_micro():
    cols = feature_columns(micro=True)
    assert cols[-len(MICRO_FEATURE_COLUMNS) :] == MICRO_FEATURE_COLUMNS
    assert feature_columns() == feature_columns(micro=False)  # default unchanged


def test_build_features_with_micro():
    feats = build_features(_ohlcv(), micro=True)
    for col in MICRO_FEATURE_COLUMNS:
        assert col in feats.columns
