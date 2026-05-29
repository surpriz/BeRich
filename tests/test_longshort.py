"""Tests for the long/short backtest and Sharpe-significance assessment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.backtest.longshort import LongShortConfig, run_longshort_backtest
from berich.backtest.significance import _norm_cdf, _norm_ppf, assess_sharpe
from berich.training.cross_sectional import CrossSectionalOof


def test_norm_helpers_roundtrip():
    assert abs(_norm_cdf(0.0) - 0.5) < 1e-9
    assert abs(_norm_ppf(0.5)) < 1e-6
    assert abs(_norm_cdf(_norm_ppf(0.975)) - 0.975) < 1e-4


def test_assess_sharpe_strong_positive_drift():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0015, 0.005, 750))  # high, steady Sharpe
    sig = assess_sharpe(rets, n_trials=10)
    assert sig.sharpe > 0
    assert sig.deflated_sharpe > 0.95
    assert sig.p_value < 0.05
    assert 0.0 <= sig.bootstrap_p_value <= 1.0


def test_assess_sharpe_no_edge():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0, 0.01, 500))  # zero drift
    sig = assess_sharpe(rets, n_trials=50)
    assert sig.deflated_sharpe < 0.95


def _prices(n_tickers: int = 12, n: int = 180, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    out = {}
    for i in range(n_tickers):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
        out[f"T{i:02d}"] = pd.DataFrame({"close": close}, index=idx)
    return out


def _oracle_oof(prices: dict[str, pd.DataFrame], horizon: int = 5) -> CrossSectionalOof:
    """Build an OOF whose score equals the realized forward return (oracle ranking)."""
    rows = []
    tickers = sorted(prices)
    fwd = {t: prices[t]["close"].shift(-horizon) / prices[t]["close"] - 1 for t in tickers}
    dates = prices[tickers[0]].index[60:-horizon]
    for d in dates:
        for t in tickers:
            score = float(fwd[t].loc[d])
            rows.append({"date": d, "score": score, "y_true": score, "ticker": t})
    frame = pd.DataFrame(rows).set_index("date").sort_index()
    return CrossSectionalOof(frame=frame)


def test_longshort_oracle_is_profitable_and_dollar_neutral():
    prices = _prices()
    oof = _oracle_oof(prices)
    cfg = LongShortConfig(
        top_decile=0.25, bottom_decile=0.25, weighting="equal", min_names=8, gross_leverage=1.0
    )
    res = run_longshort_backtest(prices, oof, cfg, n_trials=1)
    assert res.n_rebalances > 0
    assert len(res.returns) > 0
    # Ranking on realized forward return should make the long/short book profitable.
    assert res.metrics.total_return > 0
    # Dollar-neutral: average gross exposure ~ gross_leverage (long 0.5 + short 0.5).
    assert abs(res.avg_gross_exposure - 1.0) < 0.25


def test_longshort_empty_oof_is_safe():
    prices = _prices(n_tickers=4)
    empty = CrossSectionalOof(
        frame=pd.DataFrame(columns=["score", "y_true", "ticker"]).rename_axis("date")
    )
    res = run_longshort_backtest(prices, empty, LongShortConfig())
    assert res.n_rebalances == 0
    assert res.returns.empty
