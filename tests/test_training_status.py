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
        metrics={
            "auc": 0.58,
            "sharpe": 0.6,
            "benchmark_sharpe": 0.4,
            "n_trades": 30.0,
            "deflated_sharpe": 0.98,
            "sharpe_pvalue": 0.01,
        },
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


def test_study_matches_interval_isolates_intraday_from_daily():
    """An intraday (-1h-) study must NOT count toward the daily trial total, and vice-versa."""
    from berich.training.status import _hpo_trials_for, _study_matches  # noqa: PLC0415

    daily = "berich-hpo-BTC-USD-lgbm-long-auc"
    intraday = "berich-hpo-BTC-USD-lgbm-long-1h-auc"
    assert _study_matches(daily, "BTC-USD", "long", None, "fixed") is True
    assert _study_matches(intraday, "BTC-USD", "long", None, "fixed") is False
    assert _study_matches(intraday, "BTC-USD", "long", None, "fixed", "1h") is True
    assert _study_matches(daily, "BTC-USD", "long", None, "fixed", "1h") is False
    counts = {daily: 30, intraday: 100}
    assert _hpo_trials_for(counts, "BTC-USD", None, "long", "fixed") == 30
    assert _hpo_trials_for(counts, "BTC-USD", None, "long", "fixed", "1h") == 100


def test_naive_local_to_utc_iso_normalizes():
    from berich.training.status import _naive_local_to_utc_iso  # noqa: PLC0415

    assert _naive_local_to_utc_iso(None) is None
    assert _naive_local_to_utc_iso("") is None
    assert _naive_local_to_utc_iso("not-a-date") is None
    out = _naive_local_to_utc_iso("2026-06-04 09:02:28.056155")
    assert out is not None
    assert out.endswith("+00:00")  # naive local timestamp anchored and converted to UTC


def test_hpo_combo_sort_key_unsearched_first_then_oldest():
    """The continuous sweep re-fit order: un-searched combos first, then oldest-HPO-first."""
    from berich.training.status import hpo_combo_sort_key  # noqa: PLC0415

    counts = {
        "berich-hpo-AAA-lgbm-long-auc": 100,  # searched, daily fixed, older
        "berich-hpo-BBB-lgbm-long-auc": 50,  # searched, daily fixed, newer
        # CCC has no study at all -> never searched (no model yet)
    }
    times = {
        "berich-hpo-AAA-lgbm-long-auc": "2026-06-01T06:00:00+00:00",
        "berich-hpo-BBB-lgbm-long-auc": "2026-06-08T06:00:00+00:00",
    }
    k_aaa = hpo_combo_sort_key(counts, times, "AAA", "long", "fixed")
    k_bbb = hpo_combo_sort_key(counts, times, "BBB", "long", "fixed")
    k_ccc = hpo_combo_sort_key(counts, times, "CCC", "long", "fixed")

    # An asset with no model yet is effectively infinitely stale -> sorts before any searched one.
    assert k_ccc < k_aaa
    assert k_ccc < k_bbb
    # Among searched combos, the older last-HPO is re-fit first (bounds max staleness).
    assert k_aaa < k_bbb
    assert k_ccc == (False, "")
    assert k_aaa[0] is True


def test_hpo_combo_sort_key_isolates_intraday_interval():
    """An intraday combo's staleness is read from its own (-1h-) study, not the daily one."""
    from berich.training.status import hpo_combo_sort_key  # noqa: PLC0415

    counts = {"berich-hpo-BTC-USD-lgbm-long-auc": 100}  # daily only; no intraday study
    times = {"berich-hpo-BTC-USD-lgbm-long-auc": "2026-06-08T06:00:00+00:00"}
    daily = hpo_combo_sort_key(counts, times, "BTC-USD", "long", "fixed", "1d")
    intra = hpo_combo_sort_key(counts, times, "BTC-USD", "long", "fixed", "1h")
    assert daily[0] is True  # daily side has a study
    assert intra == (False, "")  # intraday side has none -> un-searched, sorts first


def test_sweep_refit_order_interleaves_incumbents():
    """Un-searched lead, but one incumbent (oldest-first) is spliced in every N new combos."""
    from berich.training.status import sweep_refit_order  # noqa: PLC0415

    new = [(f"NEW{i}", "long", "fixed", "1d") for i in range(1, 7)]  # 6 un-searched
    olda = ("OLDA", "long", "fixed", "1d")
    oldb = ("OLDB", "long", "fixed", "1d")
    counts = {"berich-hpo-OLDA-lgbm-long-auc": 100, "berich-hpo-OLDB-lgbm-long-auc": 100}
    times = {
        "berich-hpo-OLDA-lgbm-long-auc": "2026-06-01T00:00:00+00:00",  # oldest
        "berich-hpo-OLDB-lgbm-long-auc": "2026-06-05T00:00:00+00:00",
    }
    out = sweep_refit_order([*new, olda, oldb], counts, times, interleave_every=2)
    # N N <oldest> N N <next-oldest> N N
    assert out == [new[0], new[1], olda, new[2], new[3], oldb, new[4], new[5]]


def test_sweep_refit_order_degrades_to_new_then_oldest_first():
    from berich.training.status import sweep_refit_order  # noqa: PLC0415

    new = [("NEW1", "long", "fixed", "1d"), ("NEW2", "long", "fixed", "1d")]
    olda = ("OLDA", "long", "fixed", "1d")
    oldb = ("OLDB", "long", "fixed", "1d")
    counts = {"berich-hpo-OLDA-lgbm-long-auc": 100, "berich-hpo-OLDB-lgbm-long-auc": 100}
    times = {
        "berich-hpo-OLDA-lgbm-long-auc": "2026-06-01T00:00:00+00:00",
        "berich-hpo-OLDB-lgbm-long-auc": "2026-06-05T00:00:00+00:00",
    }
    # Disabled interleaving -> pure new-first, then oldest-first.
    assert sweep_refit_order([oldb, *new, olda], counts, times, interleave_every=0) == [
        *new,
        olda,
        oldb,
    ]
    # No incumbents -> just the un-searched; no un-searched -> incumbents oldest-first.
    assert sweep_refit_order(new, {}, {}, interleave_every=3) == new
    assert sweep_refit_order([oldb, olda], counts, times, interleave_every=3) == [olda, oldb]


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
