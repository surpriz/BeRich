"""Tests for point-in-time fundamental features — the no-lookahead guarantee."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features.build import build_features, feature_columns
from berich.features.fundamental_features import (
    FUNDAMENTAL_FEATURE_COLUMNS,
    PUBLICATION_LAG_DAYS,
    build_fundamental_features,
)


def _quarterly(periods: int = 8) -> pd.DataFrame:
    ends = pd.date_range("2022-03-31", periods=periods, freq="QE")
    return pd.DataFrame(
        {
            "revenue": np.linspace(1e9, 1.5e9, periods),
            "net_income": np.linspace(1e8, 2e8, periods),
            "total_assets": np.full(periods, 5e9),
            "total_equity": np.full(periods, 2e9),
            "total_debt": np.full(periods, 1e9),
        },
        index=pd.DatetimeIndex(ends, name="period_end"),
    )


def test_neutral_when_no_fundamentals():
    idx = pd.bdate_range("2023-01-01", periods=50)
    feats = build_fundamental_features(idx, None)
    assert list(feats.columns) == FUNDAMENTAL_FEATURE_COLUMNS
    assert (feats == 0.0).all().all()


def test_point_in_time_no_lookahead():
    q = _quarterly()
    idx = pd.bdate_range("2022-01-01", "2024-06-30")
    feats = build_fundamental_features(idx, q)
    first_period_end = q.index[0]
    # Before the first filing (period-end + lag) the features must still be the neutral 0.
    just_before = first_period_end + pd.Timedelta(days=PUBLICATION_LAG_DAYS - 5)
    pre = feats.loc[feats.index <= just_before]
    assert (pre == 0.0).all().all(), "fundamentals leaked before their publication date"
    # After the lag, at least one ratio is populated (non-zero).
    after = first_period_end + pd.Timedelta(days=PUBLICATION_LAG_DAYS + 10)
    row = feats.loc[feats.index >= after].iloc[0]
    assert row.abs().sum() > 0.0


def test_appending_quarter_does_not_change_earlier_values():
    q = _quarterly(8)
    idx = pd.bdate_range("2022-01-01", "2024-06-30")
    full = build_fundamental_features(idx, q)
    truncated = build_fundamental_features(idx, q.iloc[:-1])
    # Dropping the latest quarter must not change any value dated before that quarter's filing.
    cutoff = q.index[-1] + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    common = full.index[full.index < cutoff]
    pd.testing.assert_frame_equal(full.loc[common], truncated.loc[common])


def test_feature_columns_appends_fundamentals():
    cols = feature_columns(fundamentals=True)
    assert cols[-len(FUNDAMENTAL_FEATURE_COLUMNS) :] == FUNDAMENTAL_FEATURE_COLUMNS


def test_build_features_with_fundamentals():
    idx = pd.bdate_range("2022-01-01", periods=260)
    close = pd.Series(np.linspace(100, 130, 260), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(260, 1e6),
        },
        index=idx,
    )
    feats = build_features(df, fundamentals=_quarterly())
    for col in FUNDAMENTAL_FEATURE_COLUMNS:
        assert col in feats.columns
