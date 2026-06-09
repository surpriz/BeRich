"""Tests for the centralized bars-per-year annualization helper (intraday POC)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from berich.backtest.annualization import DEFAULT_BARS_PER_YEAR, bars_per_year
from berich.backtest.metrics import sharpe_ratio


def test_bars_per_year_map():
    assert bars_per_year("1d") == 252
    assert bars_per_year("1h") == 24 * 365 == 8760
    assert bars_per_year("unknown") == DEFAULT_BARS_PER_YEAR == 252


def test_sharpe_default_is_backward_compatible():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 504))
    # No kwarg must equal the explicit daily factor — the daily pipeline is unchanged.
    assert sharpe_ratio(rets) == sharpe_ratio(rets, bars_per_year=252)


def test_sharpe_scales_with_bars_per_year():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.001, 0.01, 1000))
    daily = sharpe_ratio(rets, bars_per_year=252)
    hourly = sharpe_ratio(rets, bars_per_year=8760)
    assert math.isclose(hourly, daily * math.sqrt(8760 / 252), rel_tol=1e-9)
