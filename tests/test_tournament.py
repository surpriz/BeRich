"""Tests for the per-ticker HPO study plumbing and the per-ticker tournament."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from berich.config import Config
from berich.data.store import OhlcvStore
from berich.models import LGBMModel, ModelMetadata
from berich.models.registry import ACTIVE_POINTER, META_FILE
from berich.training import tournament as tour
from berich.training.hpo import best_for_ticker, run_ticker_hpo, ticker_study_name
from berich.training.tournament import (
    CandidateResult,
    TournamentResult,
    train_candidate,
    train_ticker_tournament,
)


def _ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rets = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def _config_with_ticker(tmp_path, ticker: str = "AAA") -> Config:
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save(ticker, _ohlcv())
    store.save("SPY", _ohlcv(seed=1))
    return Config(
        data_dir=tmp_path,
        watchlist=[ticker],
        universes={"us_stocks": [ticker]},
    )


def test_ticker_study_name_is_slugged():
    assert ticker_study_name("GC=F", "lgbm", "long") == "berich-hpo-GC_F-lgbm-long-auc"


def test_run_ticker_hpo_creates_study_and_best_for_ticker(tmp_path):
    cfg = _config_with_ticker(tmp_path)
    study = run_ticker_hpo(cfg, "AAA", "lgbm", "long", n_trials=2)
    assert study.study_name == ticker_study_name("AAA", "lgbm", "long")
    assert cfg.optuna_db.exists()

    params, features = best_for_ticker(cfg, "AAA", "lgbm", "long")
    assert isinstance(params, dict)
    assert all(not k.startswith("feat_") for k in params)
    assert features is None or isinstance(features, list)


def test_best_for_ticker_missing_study(tmp_path):
    cfg = _config_with_ticker(tmp_path)
    assert best_for_ticker(cfg, "AAA", "lgbm", "long") == ({}, None)


def test_tournament_advisory_when_no_winner(tmp_path):
    cfg = _config_with_ticker(tmp_path)
    result = train_ticker_tournament(cfg, "AAA", "long", models=["lgbm"], calibrate=False)
    assert isinstance(result, TournamentResult)
    assert result.candidates  # at least the lgbm candidate ran
    # Random-walk synthetic data cannot beat buy & hold -> advisory only, nothing promoted.
    assert result.advisory_only is True
    assert result.promoted is False
    assert result.winner is None
    registry_dir = cfg.model_dir_for_ticker("AAA", "long")
    assert not (registry_dir / ACTIVE_POINTER).exists()
    # The best-AUC candidate is still saved for inspection.
    assert (registry_dir / "lgbm-long" / META_FILE).exists()


def test_tournament_promotes_when_candidate_passes(tmp_path, monkeypatch):
    cfg = _config_with_ticker(tmp_path)

    # Fabricate a gate-passing candidate deterministically (no walk-forward / backtest, whose
    # OOF probabilities depend on the global RNG state and so vary by test order). This isolates
    # the tournament's promote-the-winner logic from candidate-training nondeterminism.
    def _passing(config, store, ticker, side, model_name, *, device=None, calibrate=True):  # noqa: ARG001
        model = LGBMModel().fit(pd.DataFrame({"f": [0.0, 1.0, 0.0, 1.0]}), pd.Series([0, 1, 0, 1]))
        meta = ModelMetadata(
            name=f"{model_name}-{side}",
            framework="lightgbm",
            feature_columns=["f"],
            metrics={"auc": 0.6, "sharpe": 1.5, "benchmark_sharpe": 0.2},
            beats_buy_hold=True,
            strategy_type="long_only",
            side=side,
            ticker=ticker,
        )
        cand = CandidateResult(
            ticker=ticker,
            side=side,
            model_name=model_name,
            oos_auc=0.6,
            strategy_sharpe=1.5,
            benchmark_sharpe=0.2,
            beats_guard=True,
            framework="lightgbm",
            n_features=1,
        )
        return model, meta, None, cand

    monkeypatch.setattr(tour, "train_candidate", _passing)
    result = train_ticker_tournament(cfg, "AAA", "long", models=["lgbm"], calibrate=False)
    assert result.promoted is True
    assert result.advisory_only is False
    assert result.winner == "lgbm"

    registry_dir = cfg.model_dir_for_ticker("AAA", "long")
    pointer = registry_dir / ACTIVE_POINTER
    assert pointer.exists()
    active_name = json.loads(pointer.read_text())["name"]
    meta = ModelMetadata.model_validate_json((registry_dir / active_name / META_FILE).read_text())
    assert meta.ticker == "AAA"


def test_tournament_short_path_produces_candidate(tmp_path):
    cfg = _config_with_ticker(tmp_path)
    result = train_ticker_tournament(cfg, "AAA", "short", models=["lgbm"], calibrate=False)
    assert isinstance(result, TournamentResult)
    assert result.candidates
    cand = result.candidates[0]
    assert cand.side == "short"
    # Short metrics are significance-based; the guard rarely passes on a random walk.
    assert isinstance(cand.beats_guard, bool)


def test_tournament_skips_thin_data(tmp_path):
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save("BBB", _ohlcv(n=120))
    store.save("SPY", _ohlcv(seed=1))
    cfg = Config(data_dir=tmp_path, watchlist=["BBB"], universes={"us_stocks": ["BBB"]})
    result = train_ticker_tournament(cfg, "BBB", "long", models=["lgbm"], calibrate=False)
    assert result.advisory_only is True
    assert result.candidates == []


def test_train_candidate_returns_metadata(tmp_path):
    cfg = _config_with_ticker(tmp_path)
    store = OhlcvStore(cfg.ohlcv_dir)
    _model, meta, cal, cand = train_candidate(cfg, store, "AAA", "long", "lgbm", calibrate=True)
    assert isinstance(meta, ModelMetadata)
    assert meta.ticker == "AAA"
    assert meta.side == "long"
    assert meta.name == "lgbm-long"
    assert cand.framework == "lightgbm"
    assert cal is not None
