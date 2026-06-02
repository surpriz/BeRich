"""Tests for triple-barrier labeling on controlled synthetic series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from berich.features.volatility import VolForecast
from berich.labeling.triple_barrier import (
    LabelConfig,
    adaptive_barriers,
    triple_barrier_labels,
)


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


def _short_cfg(horizon: int = 5) -> LabelConfig:
    return LabelConfig(
        horizon_days=horizon,
        atr_window=3,
        take_profit_atr=2.0,
        stop_loss_atr=2.0,
        direction="short",
    )


def test_short_labels_one_where_long_labels_minus_one_on_falling_series():
    closes = [100, 101, 99, 100, 98, 95, 92, 89, 86, 83, 80]
    df = _series(closes)
    long_out = triple_barrier_labels(df, _cfg())["label"].dropna()
    short_out = triple_barrier_labels(df, _short_cfg())["label"].dropna()
    assert long_out.iloc[0] == -1
    assert short_out.iloc[0] == 1


def test_short_labels_minus_one_on_rising_series():
    df = _series([100, 99, 101, 100, 102, 105, 108, 111, 114, 117, 120])
    out = triple_barrier_labels(df, _short_cfg())["label"].dropna()
    assert out.iloc[0] == -1


def test_short_winning_return_is_positive():
    df = _series([100, 101, 99, 100, 98, 95, 92, 89, 86, 83, 80])
    out = triple_barrier_labels(df, _short_cfg()).dropna()
    wins = out[out["label"] == 1]
    assert len(wins) > 0
    assert (wins["ret"] > 0).all()


def test_short_both_barriers_in_bar_is_minus_one():
    idx = pd.bdate_range("2020-01-01", periods=12)
    c = np.full(12, 100.0)
    high = c.copy()
    low = c.copy()
    # Bar 4 straddles both the short stop (above) and short target (below).
    high[4] = 130.0
    low[4] = 70.0
    df = pd.DataFrame({"open": c, "high": high, "low": low, "close": c}, index=idx)
    out = triple_barrier_labels(df, _short_cfg(horizon=5))
    assert out["label"].iloc[3] == -1


def test_short_last_horizon_rows_are_nan():
    df = _series([120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110])
    out = triple_barrier_labels(df, _short_cfg(horizon=5))
    assert out["label"].iloc[-5:].isna().all()


def test_adaptive_barriers_short_mirrors_long():
    vf = VolForecast(sigma_daily=0.02, horizon_sigma=0.06, method="ewma")
    cfg = LabelConfig()
    stop_l, target_l, _ = adaptive_barriers(100.0, 2.0, vf, cfg)
    stop_s, target_s, rationale = adaptive_barriers(100.0, 2.0, vf, cfg, direction="short")
    # Long target above entry, short target below; stops mirror.
    assert target_l > 100.0 > target_s
    assert stop_l < 100.0 < stop_s
    assert rationale["direction"] == "short"


def test_adaptive_barriers_long_unchanged_with_direction_kwarg():
    vf = VolForecast(sigma_daily=0.02, horizon_sigma=0.06, method="ewma")
    cfg = LabelConfig()
    s1, t1, _ = adaptive_barriers(100.0, 2.0, vf, cfg)
    s2, t2, _ = adaptive_barriers(100.0, 2.0, vf, cfg, direction="long")
    assert s1 == s2
    assert t1 == t2


# ----------------------------------------------------------------- trailing exit ----


def _trail_cfg(mode: str = "trailing", direction: str = "long", horizon: int = 5) -> LabelConfig:
    return LabelConfig(
        horizon_days=horizon,
        atr_window=3,
        take_profit_atr=2.0,
        stop_loss_atr=2.0,
        trailing_atr=2.5,
        trailing_activation_atr=1.0,
        direction=direction,  # ty: ignore[invalid-argument-type]
        exit_mode=mode,  # ty: ignore[invalid-argument-type]
    )


def test_trailing_rides_past_fixed_take_profit():
    # A clean uptrend: the fixed TP caps the win at ~2 ATR, the trailing exit lets it run to
    # the horizon close, so the trailing realized return is strictly larger.
    df = _series([100, 99, 101, 100, 102, 105, 108, 111, 114, 117, 120])
    fixed = triple_barrier_labels(df, _cfg())["ret"].dropna().iloc[0]
    trailing = triple_barrier_labels(df, _trail_cfg())["ret"].dropna().iloc[0]
    assert trailing > fixed > 0


def test_trailing_tp_caps_at_fixed_target():
    # With the TP kept as a cap, an uptrend exits at that fixed target — same as the fixed
    # label — so trailing_tp gives back the upside the pure trailing variant captures.
    df = _series([100, 99, 101, 100, 102, 105, 108, 111, 114, 117, 120])
    fixed = triple_barrier_labels(df, _cfg())["ret"].dropna().iloc[0]
    capped = triple_barrier_labels(df, _trail_cfg(mode="trailing_tp"))["ret"].dropna().iloc[0]
    pure = triple_barrier_labels(df, _trail_cfg())["ret"].dropna().iloc[0]
    assert capped == fixed
    assert pure > capped


def test_trailing_short_rides_downtrend_and_wins():
    df = _series([100, 101, 99, 100, 98, 95, 92, 89, 86, 83, 80])
    out = triple_barrier_labels(df, _trail_cfg(direction="short")).dropna()
    assert out["label"].iloc[0] == 1
    assert out["ret"].iloc[0] > 0


def test_trailing_stop_is_causal_no_same_bar_lookahead():
    # A favorable intrabar spike (high=110) that fully retraces within the SAME bar (low=99)
    # must NOT exit on that bar — the stop is set from the PRIOR extreme. The exit only fires
    # on the next bar whose low breaks the ratcheted stop. A lookahead bug would exit one bar
    # early (bars_held==1); the causal engine exits at bars_held==2, locking +5%.
    n = 12
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close[8:] = 104.0
    high[7], low[7] = 110.0, 99.0  # bar A: spike up, full retrace
    high[8], low[8] = 106.0, 104.0  # bar B: low breaks the ratcheted stop (~105)
    high[9:], low[9:] = 105.0, 103.0
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)
    out = triple_barrier_labels(df, _trail_cfg(horizon=5))
    assert out["bars_held"].iloc[6] == 2  # exited on bar B, not the spike bar A
    assert out["label"].iloc[6] == 1
    assert out["ret"].iloc[6] == pytest.approx(0.05, abs=1e-9)
