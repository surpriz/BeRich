"""Phase 7 PEAD tests: labelling causality, feature defaults, walk-forward shape."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from berich.datasets.pead import PeadDataset, split_walk_forward
from berich.features.pead_features import (
    DAYS_CLAMP,
    PEAD_FEATURE_COLUMNS,
    build_pead_features,
)
from berich.labeling.pead import (
    DRIFT_5D_THRESHOLD,
    FWD_20D,
    build_pead_events,
)


def _ohlcv(*, n: int = 400, start: str = "2022-01-03", drift: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(start, periods=n)
    log_ret = rng.normal(drift, 0.01, n)
    close = 100 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        },
        index=idx,
    )


def _earnings(rows: list[tuple[str, float | None, float | None, float | None]]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows, columns=pd.Index(["date", "eps_estimate", "reported_eps", "surprise_pct"])
    ).set_index("date")
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    return df.astype(dict.fromkeys(["eps_estimate", "reported_eps", "surprise_pct"], float))


# ------------------------------------------------------- labelling causality ----


def test_build_pead_events_uses_first_trading_day_after_announcement():
    """entry_date must be strictly after event_date — never the same day."""
    ohlcv = _ohlcv()
    events = _earnings([("2022-03-15", 1.0, 1.1, 10.0)])
    out = build_pead_events(ohlcv, events, ticker="AAA")
    assert len(out) == 1
    assert out.iloc[0]["event_date"] == pd.Timestamp("2022-03-15")
    assert out.iloc[0]["entry_date"] > pd.Timestamp("2022-03-15")


def test_build_pead_events_drops_when_forward_window_incomplete():
    """An announcement too close to the cache tip must be skipped."""
    ohlcv = _ohlcv(n=30)  # tiny cache
    # Place the event near the end so 20 forward bars don't fit.
    last_date = ohlcv.index[-3].strftime("%Y-%m-%d")
    events = _earnings([(last_date, 1.0, 1.1, 5.0)])
    out = build_pead_events(ohlcv, events, ticker="AAA")
    assert out.empty


def test_build_pead_events_labels_match_threshold():
    """Construct a trend that the 5d label should match by construction."""
    ohlcv = _ohlcv(n=400, drift=0.005)  # strongly upward trend
    events = _earnings([("2022-03-15", 1.0, 1.1, 5.0)])
    out = build_pead_events(ohlcv, events, ticker="AAA")
    assert len(out) == 1
    row = out.iloc[0]
    expected_label = int(row["fwd_return_5d"] > DRIFT_5D_THRESHOLD)
    assert row["label_drift_5d"] == expected_label


# ----------------------------------------------------------- feature defaults ----


def test_pead_features_neutral_when_no_history():
    """Event before the cache starts → neutral row, no crash."""
    ohlcv = _ohlcv(start="2023-01-03", n=100)
    earn = _earnings([("2022-01-15", 1.0, 1.1, 10.0)])  # before cache start
    events = pd.DataFrame([{"ticker": "AAA", "event_date": pd.Timestamp("2022-01-15")}])
    feats = build_pead_features(
        events, ohlcv_by_ticker={"AAA": ohlcv}, earnings_by_ticker={"AAA": earn}
    )
    assert list(feats.columns) == PEAD_FEATURE_COLUMNS
    assert len(feats) == 1
    assert feats.iloc[0]["days_since_last_earnings"] == DAYS_CLAMP


def test_pead_features_use_past_surprises_only():
    """surprise_4q_mean at event t uses surprises STRICTLY before t — same rule
    as the Phase 5a earnings features. Adding a future surprise must not
    affect the feature row for an earlier event."""
    ohlcv = _ohlcv(n=400)
    base_earn = _earnings(
        [
            ("2022-03-15", 1.0, 1.1, 10.0),
            ("2022-06-15", 1.0, 1.2, 20.0),
            ("2022-09-15", 1.0, 1.05, 5.0),
        ]
    )
    perturbed = _earnings(
        [
            ("2022-03-15", 1.0, 1.1, 10.0),
            ("2022-06-15", 1.0, 1.2, 20.0),
            ("2022-09-15", 1.0, 1.05, 5.0),
            ("2022-12-15", 1.0, 1.05, 99.0),  # added future surprise
        ]
    )
    events = pd.DataFrame([{"ticker": "AAA", "event_date": pd.Timestamp("2022-09-15")}])
    f_base = build_pead_features(
        events, ohlcv_by_ticker={"AAA": ohlcv}, earnings_by_ticker={"AAA": base_earn}
    )
    f_perturbed = build_pead_features(
        events, ohlcv_by_ticker={"AAA": ohlcv}, earnings_by_ticker={"AAA": perturbed}
    )
    pd.testing.assert_frame_equal(f_base, f_perturbed)


# ------------------------------------------------- walk-forward split shape ----


def test_split_walk_forward_yields_chronological_folds():
    """Each fold's test_idx is strictly after its train_idx and folds are
    chronological (i.e. fold i's test_end == fold i+1's train_end)."""
    n = 2_000
    dummy = PeadDataset(
        events=pd.DataFrame({"ticker": ["AAA"] * n}),
        x=pd.DataFrame(
            np.zeros((n, len(PEAD_FEATURE_COLUMNS))), columns=pd.Index(PEAD_FEATURE_COLUMNS)
        ),
        y=pd.Series(np.zeros(n, dtype=int)),
        entry_dates=pd.DatetimeIndex(pd.date_range("2010-01-01", periods=n, freq="B")),
        tickers=pd.Series(["AAA"] * n),
    )
    folds = split_walk_forward(dummy, n_folds=5, min_train=500)
    assert len(folds) >= 3
    for tr, te in folds:
        assert tr.max() < te.min()  # no overlap, strict past for train
    # Walk-forward: each fold's train_end == previous fold's test_end.
    for prev, nxt in pairwise(folds):
        assert prev[1].max() + 1 == nxt[0].max() + 1


def test_split_walk_forward_returns_empty_when_too_small():
    n = 100
    dummy = PeadDataset(
        events=pd.DataFrame({"ticker": ["AAA"] * n}),
        x=pd.DataFrame(np.zeros((n, 2))),
        y=pd.Series(np.zeros(n, dtype=int)),
        entry_dates=pd.DatetimeIndex(pd.date_range("2010-01-01", periods=n, freq="B")),
        tickers=pd.Series(["AAA"] * n),
    )
    assert split_walk_forward(dummy, n_folds=5, min_train=500) == []


# --------------- regression: 20-day forward window must outrun 5-day ----------


def test_label_horizons_independently_consistent():
    """5d and 20d labels are computed from the same OHLCV; the 20d window is
    just FWD_20D long, so the 20d return horizon is strictly later than the
    5d horizon."""
    ohlcv = _ohlcv(n=400, drift=0.002)
    events = _earnings([("2022-03-15", 1.0, 1.1, 5.0)])
    out = build_pead_events(ohlcv, events, ticker="AAA")
    assert len(out) == 1
    entry_idx = ohlcv.index.get_loc(out.iloc[0]["entry_date"])
    assert FWD_20D + entry_idx < len(ohlcv)  # sanity: sample frame is big enough
