"""Tests for the nightly zoo-retrain selection/promotion logic (no real training)."""

from __future__ import annotations

import berich.models as models_mod
import berich.training.deep as deep_mod
from berich.config import Config
from berich.scheduler.jobs import retrain_zoo_job
from berich.training.deep import DeepTrainResult


def _result(name: str, *, sharpe: float, beats: bool) -> DeepTrainResult:
    return DeepTrainResult(
        name=name,
        oos_auc=0.52,
        strategy_sharpe=sharpe,
        benchmark_sharpe=0.5,
        baseline_sharpe=0.4,
        beats_buy_hold=beats,
        beats_baseline=beats,
        promoted=False,
    )


def test_retrain_promotes_best_passing_candidate(monkeypatch):
    config = Config(watchlist=["AAA"])
    config.zoo.enabled_models = ["lgbm", "patchtst"]

    monkeypatch.setattr(deep_mod, "baseline_sharpe", lambda _c: 0.4)

    # lgbm passes the guard with a modest Sharpe; patchtst passes with a higher one.
    sharpes = {"lgbm-D": 0.9, "patchtst-D": 1.5}

    def fake_train(_config, *, name, **_kw):
        return _result(name, sharpe=sharpes[name], beats=True)

    monkeypatch.setattr(deep_mod, "train_deep_model", fake_train)

    promoted: list[str] = []
    monkeypatch.setattr(models_mod, "promote", lambda n, **_kw: promoted.append(n))

    out = retrain_zoo_job(config, date_str="D")
    assert out["candidates"] == 2
    assert out["promoted"] == "patchtst-D"  # highest Sharpe among passing
    assert promoted == ["patchtst-D"]


def test_retrain_promotes_nothing_when_no_candidate_passes(monkeypatch):
    config = Config(watchlist=["AAA"])
    config.zoo.enabled_models = ["lgbm"]
    monkeypatch.setattr(deep_mod, "baseline_sharpe", lambda _c: 0.4)

    def fake_train(_c, *, name, **_kw):
        return _result(name, sharpe=0.1, beats=False)

    monkeypatch.setattr(deep_mod, "train_deep_model", fake_train)
    called: list[str] = []
    monkeypatch.setattr(models_mod, "promote", lambda n, **_kw: called.append(n))

    out = retrain_zoo_job(config, date_str="D")
    assert out["promoted"] == ""
    assert called == []  # guard left the active model untouched
