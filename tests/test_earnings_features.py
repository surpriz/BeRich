"""Tests for the Phase 5a earnings features: causality, defaults, idempotence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.data.earnings import EARNINGS_COLUMNS, EarningsStore
from berich.features.build import build_features, feature_columns
from berich.features.earnings_features import (
    DAYS_CLAMP,
    EARNINGS_FEATURE_COLUMNS,
    WINDOW_DAYS,
    build_earnings_features,
)


def _ohlcv(n: int = 260, seed: int = 0, start: str = "2024-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    vol = rng.integers(1_000, 10_000, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _earnings(
    rows: list[tuple[str, float | None, float | None, float | None]],
) -> pd.DataFrame:
    """Build a canonical earnings frame from ``(date_str, est, reported, surprise%)``."""
    df = pd.DataFrame(rows, columns=pd.Index(["date", *EARNINGS_COLUMNS])).set_index("date")
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    return df.astype(dict.fromkeys(EARNINGS_COLUMNS, float))


# ----------------------------------------------------------- neutral defaults ----


def test_neutral_defaults_when_no_earnings():
    """A ticker without an earnings track must get clamp-day distances + zeros."""
    df = _ohlcv()
    feats = build_earnings_features(pd.DatetimeIndex(df.index), None)
    assert list(feats.columns) == EARNINGS_FEATURE_COLUMNS
    assert (feats["days_to_next_earnings"] == DAYS_CLAMP).all()
    assert (feats["days_since_last_earnings"] == DAYS_CLAMP).all()
    assert (feats["last_surprise_pct"] == 0).all()
    assert (feats["surprise_4q_mean"] == 0).all()
    assert (feats["pre_earnings_window"] == 0).all()
    assert (feats["post_earnings_window"] == 0).all()


def test_neutral_defaults_when_empty_frame():
    df = _ohlcv()
    feats = build_earnings_features(pd.DatetimeIndex(df.index), pd.DataFrame())
    assert (feats["days_to_next_earnings"] == DAYS_CLAMP).all()
    assert (feats["last_surprise_pct"] == 0).all()


# ------------------------------------------------------------- causality core ----


def test_past_surprise_change_does_not_affect_pre_announcement_bars():
    """Perturbing the surprise of an earnings at date e must NOT change features at t <= e.

    This is the central no-leakage guarantee: a surprise reported after-close
    on day J is unknown to features at any t <= J, so changing its value can't
    move those features.
    """
    df = _ohlcv(n=260)
    e1 = _earnings(
        [
            ("2024-01-30", 1.0, 1.10, 10.0),
            ("2024-04-30", 1.0, 1.20, 20.0),
            ("2024-07-30", 1.0, 1.05, 5.0),
            ("2024-10-30", 1.0, 0.95, -5.0),
        ]
    )
    e2 = e1.copy()
    e2.loc[pd.Timestamp("2024-07-30"), "surprise_pct"] = 999.0  # perturbation

    f1 = build_earnings_features(pd.DatetimeIndex(df.index), e1)
    f2 = build_earnings_features(pd.DatetimeIndex(df.index), e2)

    perturbed_date = pd.Timestamp("2024-07-30")
    cutoff_mask = df.index <= perturbed_date
    pd.testing.assert_frame_equal(f1[cutoff_mask], f2[cutoff_mask])


def test_past_surprise_change_does_affect_later_bars():
    """Counter-test: the perturbation must show up *after* the announcement.

    Pairs with the previous test — if both pass we know the lag boundary is
    where it should be (right after the announcement), not silently shifted.
    """
    df = _ohlcv(n=260)
    e1 = _earnings(
        [
            ("2024-01-30", 1.0, 1.10, 10.0),
            ("2024-04-30", 1.0, 1.20, 20.0),
            ("2024-07-30", 1.0, 1.05, 5.0),
            ("2024-10-30", 1.0, 0.95, -5.0),
        ]
    )
    e2 = e1.copy()
    e2.loc[pd.Timestamp("2024-07-30"), "surprise_pct"] = 999.0

    f1 = build_earnings_features(pd.DatetimeIndex(df.index), e1)
    f2 = build_earnings_features(pd.DatetimeIndex(df.index), e2)

    after_mask = df.index > pd.Timestamp("2024-07-30")
    diff = (f1[after_mask] - f2[after_mask]).abs().sum().sum()
    assert diff > 0


def test_future_only_earnings_set_pre_window_but_keep_surprise_zero():
    """If only a future earnings is known, pre_window can fire but past fields stay 0."""
    df = _ohlcv(start="2024-01-02", n=15)
    e = _earnings([("2024-01-15", 1.0, np.nan, np.nan)])
    feats = build_earnings_features(pd.DatetimeIndex(df.index), e)

    # On 2024-01-08 (Mon) days_to_next = 7 -> outside the 5-day window
    early = pd.Timestamp("2024-01-08")
    if early in df.index:
        assert feats.loc[early, "pre_earnings_window"] == 0.0
    # On 2024-01-12 (Fri) days_to_next = 3 -> inside the window
    near = pd.Timestamp("2024-01-12")
    if near in df.index:
        assert feats.loc[near, "pre_earnings_window"] == 1.0
    # Past fields stay 0 *before* the announcement (no past earnings at those bars);
    # they will activate naturally for bars > 2024-01-15 once the announcement is past.
    pre_announcement = df.index <= pd.Timestamp("2024-01-15")
    assert (feats.loc[pre_announcement, "post_earnings_window"] == 0).all()
    assert (feats.loc[pre_announcement, "last_surprise_pct"] == 0).all()


def test_post_window_excludes_same_day():
    """At t == announcement date the post-window must still be 0 (after-close rule)."""
    df = _ohlcv(start="2024-01-02", n=20)
    e = _earnings([("2024-01-10", 1.0, 1.05, 5.0)])
    feats = build_earnings_features(pd.DatetimeIndex(df.index), e)
    day_of = pd.Timestamp("2024-01-10")
    next_day = pd.Timestamp("2024-01-11")
    assert feats.loc[day_of, "post_earnings_window"] == 0.0
    if next_day in feats.index:
        assert feats.loc[next_day, "post_earnings_window"] == 1.0


def test_surprise_4q_mean_uses_only_strict_past():
    """The rolling mean at date t must use only earnings strictly before t."""
    df = _ohlcv(start="2024-01-02", n=260)
    e = _earnings(
        [
            ("2024-01-30", 1.0, 1.10, 10.0),
            ("2024-04-30", 1.0, 1.20, 20.0),
            ("2024-07-30", 1.0, 1.05, 5.0),
            ("2024-10-30", 1.0, 0.95, -5.0),
        ]
    )
    feats = build_earnings_features(pd.DatetimeIndex(df.index), e)

    # Day of the first earnings: nothing in the past yet → mean is 0 (neutral).
    assert feats.loc[pd.Timestamp("2024-01-30"), "surprise_4q_mean"] == 0.0
    # The very next business day: only the first surprise counts → mean = 10.
    after_first = pd.Timestamp("2024-01-31")
    if after_first in feats.index:
        assert abs(feats.loc[after_first, "surprise_4q_mean"] - 10.0) < 1e-9
    # After all four: mean of the four = 7.5.
    after_all = pd.Timestamp("2024-10-31")
    if after_all in feats.index:
        assert abs(feats.loc[after_all, "surprise_4q_mean"] - 7.5) < 1e-9


# ----------------------------- build_features integration + ETF survives join ----


def test_build_features_appends_earnings_columns_when_supplied():
    df = _ohlcv()
    e = _earnings([("2024-04-30", 1.0, 1.05, 5.0)])
    base = build_features(df, market=_ohlcv(seed=1))
    with_earn = build_features(df, market=_ohlcv(seed=1), earnings=e)
    assert list(with_earn.columns) == feature_columns(earnings=True)
    assert list(base.columns) == feature_columns(earnings=False)
    # Base columns survive unchanged when earnings are added.
    pd.testing.assert_frame_equal(with_earn[base.columns], base)


def test_build_features_empty_earnings_yields_neutral_defaults():
    """An empty earnings frame must produce non-NaN values so ETF rows survive dropna."""
    df = _ohlcv()
    feats = build_features(df, market=_ohlcv(seed=1), earnings=pd.DataFrame())
    for col in EARNINGS_FEATURE_COLUMNS:
        assert feats[col].notna().all(), f"{col} should be non-NaN under empty earnings"


# ----------------------------------------------------- EarningsStore idempotence ----


def test_earnings_store_save_load_roundtrip_and_dedupe(tmp_path):
    store = EarningsStore(tmp_path / "earnings")
    df1 = _earnings(
        [
            ("2024-01-30", 1.0, 1.10, 10.0),
            ("2024-04-30", 1.0, 1.20, 20.0),
        ]
    )
    store.save("AAA", df1)
    loaded = store.load("AAA")
    assert loaded is not None
    assert len(loaded) == 2

    # Re-saving the same rows merges without duplicates (idempotent).
    store.save("AAA", df1)
    loaded = store.load("AAA")
    assert loaded is not None
    assert len(loaded) == 2

    # Adding a new row merges in; old rows stay.
    df2 = _earnings([("2024-07-30", 1.0, 1.05, 5.0)])
    store.save("AAA", df2)
    loaded = store.load("AAA")
    assert loaded is not None
    assert len(loaded) == 3
    assert pd.Timestamp("2024-04-30") in loaded.index


def test_earnings_store_has_any_data(tmp_path):
    store = EarningsStore(tmp_path / "earnings")
    assert store.has_any_data() is False
    store.save("AAA", pd.DataFrame())  # cache an empty file
    assert store.has_any_data() is False
    store.save("BBB", _earnings([("2024-01-30", 1.0, 1.1, 10.0)]))
    assert store.has_any_data() is True


# ---------------------------------------------- belt-and-suspenders window check ----


def test_pre_window_clamp_boundary():
    """pre_earnings_window should fire at exactly ``WINDOW_DAYS`` calendar days out."""
    df = _ohlcv(start="2024-01-02", n=30)
    e = _earnings([(f"2024-01-{15:02d}", 1.0, np.nan, np.nan)])
    feats = build_earnings_features(pd.DatetimeIndex(df.index), e)
    # At exactly WINDOW_DAYS away (calendar days, not business days), flag must be on.
    edge = pd.Timestamp("2024-01-15") - pd.Timedelta(days=WINDOW_DAYS)
    if edge in feats.index:
        assert feats.loc[edge, "pre_earnings_window"] == 1.0
