"""Tests for the Optuna HPO objective (no GPU, tiny synthetic dataset)."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from berich.datasets.assemble import build_dataset
from berich.labeling.triple_barrier import LabelConfig
from berich.training.hpo import objective_for


def _ohlcv(seed: int, n: int = 420) -> pd.DataFrame:
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


def _dataset_and_prices():
    frames = {"AAA": _ohlcv(1), "BBB": _ohlcv(2), "SPY": _ohlcv(3)}
    store = _FakeStore(frames)
    label_cfg = LabelConfig()
    ds = build_dataset(store, ["AAA", "BBB"], label_cfg, market_ticker="SPY")
    prices = {"AAA": frames["AAA"], "BBB": frames["BBB"]}
    return ds, prices, label_cfg


def test_objective_returns_float_with_fixed_trial():
    ds, prices, label_cfg = _dataset_and_prices()
    objective = objective_for("lgbm", ds, label_cfg, prices, search_features=False)
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "reg_lambda": 1.0,
        }
    )
    sharpe = objective(trial)
    assert isinstance(sharpe, float)
    assert np.isfinite(sharpe)


def test_small_study_optimizes_and_records_attrs():
    ds, prices, label_cfg = _dataset_and_prices()
    objective = objective_for("lgbm", ds, label_cfg, prices)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)
    assert isinstance(study.best_value, float)
    assert "sharpe" in study.best_trial.user_attrs


def test_feature_search_records_selected_features():
    ds, prices, label_cfg = _dataset_and_prices()
    objective = objective_for("lgbm", ds, label_cfg, prices, search_features=True)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)
    # The search toggles feature families and records the chosen subset.
    feats = study.best_trial.user_attrs.get("features")
    assert feats is None or isinstance(feats, list)
    assert any(k.startswith("feat_") for k in study.best_trial.params)
