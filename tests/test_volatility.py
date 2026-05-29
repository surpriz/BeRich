"""Tests for the volatility forecaster and adaptive SL/TP barriers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.features.volatility import VolForecast, forecast_vol
from berich.labeling.triple_barrier import LabelConfig, adaptive_barriers


def test_forecast_vol_positive_and_horizon_scaled():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    vf = forecast_vol(close, horizon_days=10, method="ewma")
    assert vf.method == "ewma"
    assert vf.sigma_daily > 0
    assert abs(vf.horizon_sigma - vf.sigma_daily * np.sqrt(10)) < 1e-9


def test_forecast_vol_garch_falls_back_to_ewma_when_unavailable():
    # arch is not installed in this env, so method='garch' must degrade gracefully.
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 200))))
    vf = forecast_vol(close, horizon_days=5, method="garch")
    assert vf.method in {"ewma", "garch"}
    assert vf.sigma_daily > 0


def test_forecast_vol_too_short_returns_zero():
    vf = forecast_vol(pd.Series([100.0]), horizon_days=10)
    assert vf.sigma_daily == 0.0


def test_adaptive_barriers_vol_scaled_orders_correctly():
    cfg = LabelConfig(take_profit_atr=2.0, stop_loss_atr=1.0)
    vf = VolForecast(sigma_daily=0.03, horizon_sigma=0.03 * np.sqrt(10), method="ewma")
    stop, target, rationale = adaptive_barriers(entry=100.0, atr_t=2.0, vol_forecast=vf, config=cfg)
    assert stop < 100.0 < target
    assert rationale["method"] == "vol_scaled"
    assert 0.5 <= rationale["scale"] <= 2.5


def test_adaptive_barriers_quantile_path():
    cfg = LabelConfig()
    vf = VolForecast(0.0, 0.0, "ewma")
    stop, target, rationale = adaptive_barriers(
        entry=100.0, atr_t=2.0, vol_forecast=vf, config=cfg, quantiles=(-0.04, 0.06)
    )
    assert abs(target - 106.0) < 1e-9
    assert abs(stop - 96.0) < 1e-9
    assert rationale["method"] == "quantile"


def test_adaptive_scale_widens_with_higher_vol():
    cfg = LabelConfig(take_profit_atr=2.0, stop_loss_atr=1.0)
    low = adaptive_barriers(100.0, 2.0, VolForecast(0.01, 0.0, "ewma"), cfg)[2]["scale"]
    high = adaptive_barriers(100.0, 2.0, VolForecast(0.05, 0.0, "ewma"), cfg)[2]["scale"]
    assert high > low
