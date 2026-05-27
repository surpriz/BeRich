"""Tests for triple-barrier labeling on controlled synthetic series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.labeling.triple_barrier import LabelConfig, triple_barrier_labels


def _series(closes: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=len(closes))
    c = np.array(closes, dtype=float)
    # High/low straddle close tightly so the close path drives barrier touches.
    return pd.DataFrame({"open": c, "high": c + 0.01, "low": c - 0.01, "close": c}, index=idx)


def _cfg(horizon: int = 5) -> LabelConfig:
    return LabelConfig(horizon_days=horizon, atr_window=3, take_profit_atr=2.0, stop_loss_atr=2.0)


def test_rising_series_labels_up():
    # Steady climb after a small wiggle to warm up ATR → upper barrier hit first.
    df = _series([100, 99, 101, 100, 102, 105, 108, 111, 114, 117, 120])
    out = triple_barrier_labels(df, _cfg())
    early = out["label"].dropna().iloc[0]
    assert early == 1


def test_falling_series_labels_down():
    df = _series([100, 101, 99, 100, 98, 95, 92, 89, 86, 83, 80])
    out = triple_barrier_labels(df, _cfg())
    early = out["label"].dropna().iloc[0]
    assert early == -1


def test_last_horizon_rows_are_nan():
    df = _series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    out = triple_barrier_labels(df, _cfg(horizon=5))
    # Final `horizon` rows lack a full forward window → NaN labels.
    assert out["label"].iloc[-5:].isna().all()
