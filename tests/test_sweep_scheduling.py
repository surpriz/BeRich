"""Tests for the GPU-efficiency scheduling: targeted top-ups and per-fold HPO pruning."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import optuna
import pandas as pd
import pytest

from berich.config import Config
from berich.datasets.assemble import SupervisedDataset, build_dataset
from berich.labeling.triple_barrier import LabelConfig
from berich.models.registry import served_model_name
from berich.scheduler.jobs import _topup_models
from berich.training.hpo import objective_for
from berich.training.walk_forward import oof_predict

WEEKDAY = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)  # Wednesday
WEEKEND = datetime(2026, 6, 13, 12, 0, tzinfo=UTC)  # Saturday


def test_served_model_name_parses_pointer(tmp_path):
    assert served_model_name(tmp_path) is None
    (tmp_path / "active.json").write_text(json.dumps({"name": "tft-short"}))
    assert served_model_name(tmp_path) == "tft"
    (tmp_path / "active.json").write_text(json.dumps({"name": "lgbm-long"}))
    assert served_model_name(tmp_path) == "lgbm"


def _config_with_winner(tmp_path, winner: str | None) -> Config:
    config = Config(data_dir=tmp_path, watchlist=["AAA"])
    if winner is not None:
        registry = config.model_dir_for_ticker("AAA", "long", "fixed")
        registry.mkdir(parents=True, exist_ok=True)
        (registry / "active.json").write_text(json.dumps({"name": f"{winner}-long"}))
    return config


def test_topup_models_weekday_narrows_to_winner_plus_lgbm(tmp_path):
    config = _config_with_winner(tmp_path, "tft")
    assert _topup_models(config, "AAA", "long", "fixed", when=WEEKDAY) == ["lgbm", "tft"]


def test_topup_models_weekend_reopens_full_contest(tmp_path):
    config = _config_with_winner(tmp_path, "tft")
    assert _topup_models(config, "AAA", "long", "fixed", when=WEEKEND) == list(
        config.zoo.ticker_tournament_models
    )


def test_topup_models_no_winner_keeps_full_zoo(tmp_path):
    config = _config_with_winner(tmp_path, None)
    assert _topup_models(config, "AAA", "long", "fixed", when=WEEKDAY) == list(
        config.zoo.ticker_tournament_models
    )


def test_topup_models_flag_off_keeps_full_zoo(tmp_path):
    config = _config_with_winner(tmp_path, "tft")
    config.zoo.topup_winner_only = False
    assert _topup_models(config, "AAA", "long", "fixed", when=WEEKDAY) == list(
        config.zoo.ticker_tournament_models
    )


def test_topup_models_lgbm_winner_dedupes(tmp_path):
    config = _config_with_winner(tmp_path, "lgbm")
    assert _topup_models(config, "AAA", "long", "fixed", when=WEEKDAY) == ["lgbm"]


# ------------------------------------------------------------------ fold callback ----


class _CoinFlip:
    """Tiny Model-protocol stub: constant 0.5 proba, instant fit."""

    def fit(self, _x, _y, sample_weight=None, tickers=None):  # noqa: ARG002
        return self

    def predict_proba(self, x, tickers=None):  # noqa: ARG002
        return np.full(len(x), 0.5)


def _tiny_dataset(n: int = 300):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return SupervisedDataset(
        x=pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)}, index=dates),
        y=pd.Series(rng.integers(0, 2, n), index=dates),
        weight=pd.Series(1.0, index=dates),
        tickers=pd.Series("AAA", index=dates),
        dates=pd.DatetimeIndex(dates),
    )


def test_oof_predict_invokes_fold_callback_per_non_final_fold():
    ds = _tiny_dataset()
    seen: list[tuple[int, int]] = []
    oof = oof_predict(
        ds,
        _CoinFlip,
        embargo=5,
        fold_callback=lambda i, partial: seen.append((i, len(partial))),
    )
    assert len(seen) >= 1
    assert [i for i, _ in seen] == list(range(len(seen)))  # fold indices in order
    sizes = [s for _, s in seen]
    assert sizes == sorted(sizes)  # the partial OOF frame accumulates
    assert sizes[-1] < len(oof.frame)  # the final fold never fires the callback


def test_oof_predict_fold_callback_exception_propagates():
    ds = _tiny_dataset()

    def _prune(_i, _partial):
        msg = "pruned"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="pruned"):
        oof_predict(ds, _CoinFlip, embargo=5, fold_callback=_prune)


def test_objective_with_pruning_reports_intermediate_values():
    # Reuse the HPO objective end-to-end with a real (in-memory) Optuna study: with
    # pruning=True every non-final fold must report a partial AUC to the trial.
    rng = np.random.default_rng(1)
    n = 420
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e6, 5e6, n).astype(float),
        },
        index=idx,
    )

    class _Store:
        def load(self, _ticker):
            return frame

    label_cfg = LabelConfig()
    ds = build_dataset(_Store(), ["AAA"], label_cfg, market_ticker="SPY")
    objective = objective_for(
        "lgbm",
        ds,
        label_cfg,
        {"AAA": frame},
        search_features=False,
        pruning=True,
    )
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
    study.optimize(objective, n_trials=1)
    assert study.trials[0].intermediate_values  # at least one fold reported
