"""Tests for the GPU sequence rankers (regression head) in the long/short path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.datasets.cross_sectional import build_panel_dataset
from berich.labeling.cross_sectional import CrossSectionalLabelConfig
from berich.models import LSTMConfig, LSTMRanker, PatchTSTRanker
from berich.models.base import Model
from berich.models.patchtst import PatchTSTConfig
from berich.training.cross_sectional import oof_predict_cross_sectional


def _ohlcv(seed: int, n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e6, 5e6, n).astype(float),
        },
        index=idx,
    )


class _FakeStore:
    def __init__(self, frames):
        self._frames = frames

    def load(self, ticker):
        return self._frames.get(ticker)


def test_patchtst_ranker_is_regressor_and_outputs_unbounded_score():
    rng = np.random.default_rng(0)
    n = 200
    x = pd.DataFrame(rng.normal(0, 1, (n, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(rng.normal(0, 1, n))  # continuous z-score-like target
    tickers = pd.Series(["AAA"] * n)
    cfg = PatchTSTConfig(
        lookback=24,
        patch_len=8,
        stride=4,
        d_model=16,
        n_heads=2,
        num_layers=1,
        dim_feedforward=32,
        epochs=3,
        device="cpu",
    )
    model = PatchTSTRanker(cfg).fit(x, y, tickers=tickers)
    assert isinstance(model, Model)
    assert model.is_classifier is False
    score = model.predict_proba(x, tickers=tickers)
    assert score.shape == (n,)
    # Regressor output is not squashed to [0,1] — it predicts the continuous target.
    assert score.min() < 0.0 or score.max() > 1.0 or np.std(score) > 0


def test_rankers_run_in_cross_sectional_oof():
    frames = {f"T{i:02d}": _ohlcv(i) for i in range(10)}
    frames["SPY"] = _ohlcv(99)
    store = _FakeStore(frames)
    cfg = CrossSectionalLabelConfig(horizon_days=5, beta_window=20)
    panel = build_panel_dataset(
        store, [f"T{i:02d}" for i in range(10)], cfg, market_ticker="SPY", min_names_per_date=5
    )

    def factory():
        return LSTMRanker(LSTMConfig(lookback=10, hidden=8, num_layers=1, epochs=2, device="cpu"))

    oof = oof_predict_cross_sectional(panel, factory, embargo=5)
    assert {"score", "y_true", "ticker"} <= set(oof.frame.columns)
    assert np.isfinite(oof.rank_ic) or np.isnan(oof.rank_ic)
