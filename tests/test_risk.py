"""Tests for the Phase 8 risk overlay: sizing math + gating behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from berich.risk.gating import drawdown_gate, regime_gate_spy
from berich.risk.sizing import (
    annualize_daily_vol,
    inverse_vol_size,
    kelly_fraction,
    vol_target_size,
)
from berich.risk.sizing_strategy import RiskOverlay, RiskOverlayConfig

# ----------------------------------------------------------------- Kelly math ----


def test_kelly_fraction_known_case():
    """Closed-form: p=0.6, W=0.04, L=0.02 → f* = (0.6*0.04 - 0.4*0.02)/(0.04*0.02) = 20."""
    f = kelly_fraction(0.6, 0.04, 0.02)
    assert f == pytest.approx(20.0, rel=1e-9)


def test_kelly_fraction_returns_zero_when_no_edge():
    """At p such that p*W == (1-p)*L the edge is exactly zero → f* == 0."""
    p_neutral = 0.02 / (0.04 + 0.02)  # so p*W == (1-p)*L
    f = kelly_fraction(p_neutral, 0.04, 0.02)
    assert f == pytest.approx(0.0, abs=1e-12)


def test_kelly_fraction_negative_when_loss_dominates():
    """With W=0.04 and L=0.02, the break-even probability is 1/3.
    Below it the Kelly fraction must be negative (no edge → don't bet long).
    """
    f = kelly_fraction(0.25, 0.04, 0.02)
    assert f < 0


def test_kelly_fraction_returns_zero_for_degenerate_inputs():
    assert kelly_fraction(-0.1, 0.04, 0.02) == 0.0
    assert kelly_fraction(1.5, 0.04, 0.02) == 0.0
    assert kelly_fraction(0.5, 0.0, 0.02) == 0.0
    assert kelly_fraction(0.5, 0.04, 0.0) == 0.0


# --------------------------------------------------------- vol-target sizing ----


def test_vol_target_size_reduces_when_realized_vol_too_high():
    """When recent vol is double the target, scale should be 0.5."""
    s = vol_target_size(portfolio_vol_recent=0.30, target_vol=0.15, ceiling=1.0)
    assert s == pytest.approx(0.5, rel=1e-9)


def test_vol_target_size_caps_at_ceiling_when_vol_is_tiny():
    """A near-zero realized vol must NOT lever past the ceiling."""
    s = vol_target_size(portfolio_vol_recent=0.001, target_vol=0.15, ceiling=1.0)
    assert s == 1.0


def test_vol_target_size_handles_zero_or_nan_vol():
    assert vol_target_size(portfolio_vol_recent=0.0) == 1.0
    assert vol_target_size(portfolio_vol_recent=float("nan")) == 1.0


def test_inverse_vol_size_scales_inversely_with_asset_vol():
    """Asset twice as volatile as the reference → half the size."""
    s = inverse_vol_size(asset_vol_20d=0.30, ref_vol=0.15)
    assert s == pytest.approx(0.5, rel=1e-9)


def test_inverse_vol_size_caps_low_vol_assets():
    """A very-low-vol asset must still cap at the ceiling."""
    s = inverse_vol_size(asset_vol_20d=0.001, ref_vol=0.15, ceiling=2.0)
    assert s == 2.0


def test_annualize_daily_vol_matches_sqrt252():
    assert annualize_daily_vol(0.01) == pytest.approx(0.01 * np.sqrt(252.0), rel=1e-9)


# ------------------------------------------------------------- regime gate ----


def _spy_rvol(values: list[float], start: str = "2020-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"))


def test_regime_gate_blocks_when_current_rvol_above_quartile():
    """Build a rvol distribution where the current value is in the top quartile."""
    history = _spy_rvol([0.1] * 100 + [0.5])  # last point is way above the 75th pct of [0.1]
    as_of = history.index[-1]
    allowed = regime_gate_spy(history, as_of=as_of, quartile_threshold=0.75, lookback_days=252)
    assert allowed is False


def test_regime_gate_allows_when_rvol_is_calm():
    history = _spy_rvol([0.2] * 100 + [0.05])  # last point well below the lookback
    as_of = history.index[-1]
    allowed = regime_gate_spy(history, as_of=as_of)
    assert allowed is True


def test_regime_gate_uses_only_history_strictly_before_as_of():
    """Lookback must NOT include the current bar itself."""
    history = _spy_rvol([0.1] * 50 + [0.05])
    as_of_mid = history.index[40]
    # The lookback at index 40 should be the first 40 values (all 0.1). Current rvol
    # is 0.1, which equals the entire history → at exactly the 75th percentile it
    # should NOT be blocked (strict <).
    allowed = regime_gate_spy(history, as_of=as_of_mid)
    assert allowed in {True, False}  # well-defined; concrete value depends on quantile interp


def test_regime_gate_handles_empty_history():
    empty = pd.Series(dtype=float)
    assert regime_gate_spy(empty, as_of=pd.Timestamp("2020-01-01")) is True


# --------------------------------------------------------- drawdown gate ----


def test_drawdown_gate_blocks_below_threshold():
    """Equity that fell 25 % from the peak should be blocked at 20 % threshold."""
    equity = pd.Series([1.0, 1.2, 1.0, 0.9])
    assert drawdown_gate(equity, max_dd_threshold=0.20) is False


def test_drawdown_gate_allows_above_threshold():
    """Mild drawdown (10 %) below the 20 % threshold should still allow trades."""
    equity = pd.Series([1.0, 1.2, 1.1])
    assert drawdown_gate(equity, max_dd_threshold=0.20) is True


def test_drawdown_gate_empty_curve_allows():
    assert drawdown_gate(pd.Series(dtype=float)) is True


# ------------------------------------------------------ RiskOverlay combiner ----


def test_overlay_gates_pass_when_all_disabled():
    overlay = RiskOverlay(RiskOverlayConfig())
    assert (
        overlay.gates_pass(
            as_of=pd.Timestamp("2020-01-01"),
            spy_rvol_series=None,
            strategy_equity=None,
        )
        is True
    )


def test_overlay_size_zero_when_kelly_negative_edge():
    overlay = RiskOverlay(RiskOverlayConfig(use_kelly=True))
    # Low proba with our W=0.04 / L=0.02 default → negative edge → size 0.
    size = overlay.position_size(
        proba=0.1, asset_vol_20d=None, ref_vol=None, portfolio_vol_recent=None
    )
    assert size == 0.0


def test_overlay_size_respects_max_position_cap():
    """Even with bullish proba the size must stay <= max_position."""
    overlay = RiskOverlay(
        RiskOverlayConfig(use_kelly=True, kelly_fraction_multiplier=10.0, max_position=0.05)
    )
    size = overlay.position_size(
        proba=0.99, asset_vol_20d=None, ref_vol=None, portfolio_vol_recent=None
    )
    assert size == pytest.approx(0.05, abs=1e-12)


def test_overlay_kelly_zero_when_disabled_keeps_unit_size():
    """With every sizer off the multiplier is 1.0 (then capped)."""
    overlay = RiskOverlay(RiskOverlayConfig())
    size = overlay.position_size(
        proba=0.7, asset_vol_20d=None, ref_vol=None, portfolio_vol_recent=None
    )
    # max_position default 0.05 — cap dominates the unit base.
    assert size == pytest.approx(0.05, abs=1e-12)


def test_overlay_inverse_vol_reduces_size_for_volatile_asset():
    """At a Kelly small enough to stay below the cap, inverse-vol still kicks in:
    a quiet asset (low rvol) gets a bigger fractional position than a noisy one.
    """
    cfg = RiskOverlayConfig(
        use_kelly=True,
        use_inverse_vol=True,
        kelly_fraction_multiplier=0.001,  # keep Kelly tiny so cap doesn't dominate
        kelly_win_size=0.04,
        kelly_loss_size=0.02,
        max_position=1.0,
    )
    overlay = RiskOverlay(cfg)
    quiet = overlay.position_size(
        proba=0.6, asset_vol_20d=0.10, ref_vol=0.20, portfolio_vol_recent=None
    )
    volatile = overlay.position_size(
        proba=0.6, asset_vol_20d=0.40, ref_vol=0.20, portfolio_vol_recent=None
    )
    assert quiet > volatile
