"""Tests for the per-asset training-status scanner backing /api/training."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from berich.config import AssetUniverses, Config
from berich.models import LGBMModel, ModelMetadata, promote, save_model
from berich.training.status import training_status


def _cfg(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path,
        universes=AssetUniverses(us_stocks=["AAA"], forex=["EURUSD=X"]),
    )


def _model() -> LGBMModel:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(0, 1, (120, 3)), columns=["a", "b", "c"])
    y = pd.Series((x["a"] + rng.normal(0, 0.1, 120) > 0).astype(int))
    return LGBMModel(n_estimators=10).fit(x, y)


def test_never_trained_when_no_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    rows = training_status(cfg)
    # 2 tickers x 2 sides.
    assert len(rows) == 4
    assert {r["ticker"] for r in rows} == {"AAA", "EURUSD=X"}
    assert all(r["status"] == "never_trained" for r in rows)
    assert all(r["hpo_trials"] == 0 for r in rows)


def test_optimized_only_filters_to_assets_with_hpo(tmp_path):
    import optuna  # noqa: PLC0415

    from berich.training.hpo import ticker_study_name  # noqa: PLC0415

    cfg = _cfg(tmp_path)
    # Give AAA an HPO study (1 trial); EURUSD has none.
    study = optuna.create_study(
        study_name=ticker_study_name("AAA", "lgbm", "long"),
        storage=f"sqlite:///{cfg.optuna_db}",
        direction="maximize",
        load_if_exists=True,
    )
    study.add_trial(optuna.trial.create_trial(value=0.6, params={}, distributions={}))

    full = training_status(cfg)
    opt = training_status(cfg, optimized_only=True)
    assert {r["ticker"] for r in full} == {"AAA", "EURUSD=X"}
    assert {r["ticker"] for r in opt} == {"AAA"}  # only the HPO'd asset


def test_promoted_and_advisory_are_reported(tmp_path):
    cfg = _cfg(tmp_path)
    # AAA long: promoted (beats buy & hold).
    reg = cfg.model_dir_for_ticker("AAA", "long")
    meta = ModelMetadata(
        name="tft-long",
        framework="tft",
        feature_columns=["a", "b", "c"],
        metrics={"auc": 0.58, "sharpe": 0.6, "benchmark_sharpe": 0.4, "n_trades": 30.0},
        beats_buy_hold=True,
        ticker="AAA",
        side="long",
    )
    save_model(_model(), meta, registry_dir=reg)
    promote("tft-long", registry_dir=reg)

    # EURUSD long: advisory-only (saved candidate, not promoted).
    reg2 = cfg.model_dir_for_ticker("EURUSD=X", "long")
    meta2 = ModelMetadata(
        name="lgbm-long",
        framework="lightgbm",
        feature_columns=["a", "b", "c"],
        metrics={"auc": 0.52, "sharpe": 0.3, "benchmark_sharpe": 0.9},
        beats_buy_hold=False,
        ticker="EURUSD=X",
        side="long",
    )
    save_model(_model(), meta2, registry_dir=reg2)  # saved but never promoted

    rows = {(r["ticker"], r["side"]): r for r in training_status(cfg)}
    promoted = rows[("AAA", "long")]
    assert promoted["status"] == "promoted"
    assert promoted["winner"] == "tft-long"
    assert promoted["framework"] == "tft"
    assert promoted["metrics"]["auc"] == 0.58

    advisory = rows[("EURUSD=X", "long")]
    assert advisory["status"] == "advisory_only"
    assert advisory["winner"] is None
    assert advisory["framework"] == "lightgbm"


def test_status_json_candidates_surface(tmp_path):
    cfg = _cfg(tmp_path)
    reg = cfg.model_dir_for_ticker("AAA", "short")
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "status.json").write_text(
        json.dumps(
            {
                "ticker": "AAA",
                "side": "short",
                "winner": None,
                "promoted": False,
                "advisory_only": True,
                "trained_at": "2026-06-01T06:00:00+00:00",
                "candidates": [
                    {"model_name": "lgbm", "oos_auc": 0.51, "beats_guard": False},
                    {"model_name": "lstm", "oos_auc": 0.53, "beats_guard": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = {(r["ticker"], r["side"]): r for r in training_status(cfg)}
    short = rows[("AAA", "short")]
    assert short["status"] == "advisory_only"
    assert len(short["candidates"]) == 2
    assert short["trained_at"] == "2026-06-01T06:00:00+00:00"


def test_naive_local_to_utc_iso_normalizes():
    from berich.training.status import _naive_local_to_utc_iso  # noqa: PLC0415

    assert _naive_local_to_utc_iso(None) is None
    assert _naive_local_to_utc_iso("") is None
    assert _naive_local_to_utc_iso("not-a-date") is None
    out = _naive_local_to_utc_iso("2026-06-04 09:02:28.056155")
    assert out is not None
    assert out.endswith("+00:00")  # naive local timestamp anchored and converted to UTC


def test_last_hpo_at_surfaces_latest_trial_time(tmp_path):
    import optuna  # noqa: PLC0415

    from berich.training.hpo import ticker_study_name  # noqa: PLC0415

    cfg = _cfg(tmp_path)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=ticker_study_name("AAA", "lgbm", "long"),
        storage=f"sqlite:///{cfg.optuna_db}",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)

    rows = {(r["ticker"], r["side"]): r for r in training_status(cfg)}
    aaa_long = rows[("AAA", "long")]
    assert aaa_long["last_hpo_at"] is not None
    assert aaa_long["last_hpo_at"].endswith("+00:00")  # ISO-UTC like trained_at
    # An asset with no HPO study has no timestamp.
    assert rows[("EURUSD=X", "long")]["last_hpo_at"] is None
