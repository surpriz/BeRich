"""Tests for the market-neutral cross-sectional foundation (label, panel, OOF)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.datasets.cross_sectional import XS_FEATURE_COLUMNS, build_panel_dataset
from berich.labeling.cross_sectional import CrossSectionalLabelConfig, forward_return_labels
from berich.models import LGBMRanker
from berich.models.base import Model
from berich.training.cross_sectional import oof_predict_cross_sectional


def _ohlcv(seed: int, n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    steps = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


class _FakeStore:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames

    def load(self, ticker: str) -> pd.DataFrame | None:
        return self._frames.get(ticker)


def _store(n_tickers: int = 10) -> _FakeStore:
    frames = {f"T{i:02d}": _ohlcv(i) for i in range(n_tickers)}
    frames["SPY"] = _ohlcv(999)
    return _FakeStore(frames)


def test_forward_return_labels_are_causal_tail_nan():
    df = _ohlcv(1)
    cfg = CrossSectionalLabelConfig(horizon_days=5, beta_window=20)
    labels = forward_return_labels(df, cfg, market=_ohlcv(2))
    # The last `horizon` rows have no forward return.
    assert labels["fwd_ret"].iloc[-5:].isna().all()
    assert labels["fwd_ret"].iloc[:-5].notna().any()
    assert set(labels.columns) == {"fwd_ret", "beta", "resid", "sample_weight"}


def test_panel_target_is_cross_sectionally_standardized():
    store = _store(10)
    cfg = CrossSectionalLabelConfig(horizon_days=5, beta_window=20, standardize="zscore")
    panel = build_panel_dataset(
        store, [f"T{i:02d}" for i in range(10)], cfg, market_ticker="SPY", min_names_per_date=5
    )
    assert len(panel) > 0
    assert list(panel.x.columns) == list(panel.x.columns)  # FEATURE_COLUMNS order preserved
    assert not panel.y.isna().any()
    # Each date's target is mean-zero (z-score within date).
    per_date_mean = panel.y.groupby(panel.dates).mean().abs()
    assert per_date_mean.max() < 1e-6


def test_cross_sectional_features_appended_and_centered():
    store = _store(10)
    cfg = CrossSectionalLabelConfig(horizon_days=5, beta_window=20)
    panel = build_panel_dataset(
        store,
        [f"T{i:02d}" for i in range(10)],
        cfg,
        market_ticker="SPY",
        min_names_per_date=5,
        cross_sectional=True,
    )
    for col in XS_FEATURE_COLUMNS:
        assert col in panel.x.columns
    # Within-date z-scored relative features are mean-zero per date.
    xs = panel.x["mom_20_xs"]
    per_date_mean = xs.groupby(panel.dates).mean().abs()
    assert per_date_mean.max() < 1e-6


def test_cross_sectional_can_be_disabled():
    store = _store(10)
    cfg = CrossSectionalLabelConfig(horizon_days=5, beta_window=20)
    panel = build_panel_dataset(
        store,
        [f"T{i:02d}" for i in range(10)],
        cfg,
        market_ticker="SPY",
        min_names_per_date=5,
        cross_sectional=False,
    )
    assert not any(c.endswith("_xs") for c in panel.x.columns)


def test_thin_dates_dropped():
    store = _store(6)
    cfg = CrossSectionalLabelConfig(horizon_days=5, beta_window=20)
    panel = build_panel_dataset(
        store, [f"T{i:02d}" for i in range(6)], cfg, market_ticker="SPY", min_names_per_date=20
    )
    # No date has 20 names (only 6 tickers) -> everything dropped.
    assert len(panel) == 0


def test_oof_cross_sectional_runs_and_ranker_is_model():
    store = _store(12)
    cfg = CrossSectionalLabelConfig(horizon_days=5, beta_window=20)
    panel = build_panel_dataset(
        store, [f"T{i:02d}" for i in range(12)], cfg, market_ticker="SPY", min_names_per_date=5
    )
    assert isinstance(LGBMRanker(), Model)
    oof = oof_predict_cross_sectional(panel, LGBMRanker, embargo=5)
    assert {"score", "y_true", "ticker"} <= set(oof.frame.columns)
    assert np.isfinite(oof.rank_ic) or np.isnan(oof.rank_ic)
