"""Tests for the per-ticker scheduler jobs and the CLI tournament branch (no network/GPU)."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from berich.cli import _cmd_train
from berich.config import Config
from berich.data.store import OhlcvStore
from berich.models import ModelMetadata, save_model
from berich.models.lightgbm_model import LGBMModel
from berich.models.registry import promote
from berich.scheduler import jobs as jobs_mod
from berich.scheduler.jobs import ticker_initial_sweep_job, ticker_nightly_refresh_job
from berich.training.tournament import TournamentResult


def _ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
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


def _config(tmp_path, ticker: str = "AAA") -> Config:
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save(ticker, _ohlcv())
    store.save("SPY", _ohlcv(seed=1))
    cfg = Config(data_dir=tmp_path, watchlist=[ticker], universes={"us_stocks": [ticker]})
    # Keep the sweep tiny + deterministic: one side, one strategy, lgbm only (no deep GPU models).
    cfg.zoo.ticker_sides = ["long"]
    cfg.zoo.ticker_tournament_models = ["lgbm"]
    cfg.zoo.ticker_exit_strategies = ["fixed"]
    return cfg


def _promote_stub(tmp_path, ticker: str, side: str) -> None:
    """Force a promoted per-ticker artifact so the nightly job treats the ticker as live."""
    cfg = Config(data_dir=tmp_path)
    registry_dir = cfg.model_dir_for_ticker(ticker, side)
    model = LGBMModel().fit(pd.DataFrame({"f": [0.0, 1.0, 0.0, 1.0]}), pd.Series([0, 1, 0, 1]))
    meta = ModelMetadata(
        name=f"lgbm-{side}",
        framework="lightgbm",
        feature_columns=["f"],
        metrics={"auc": 0.6, "sharpe": 1.5, "benchmark_sharpe": 0.2},
        beats_buy_hold=True,
        strategy_type="long_only",
        side=side,
        ticker=ticker,
    )
    save_model(model, meta, registry_dir=registry_dir)
    promote(meta.name, registry_dir=registry_dir)


def _stub_tournament(*, promoted: bool):
    def _fn(config, ticker, side="long", **_kw):  # noqa: ARG001
        return TournamentResult(
            ticker=ticker,
            side=side,
            candidates=[],
            winner="lgbm" if promoted else None,
            promoted=promoted,
            advisory_only=not promoted,
        )

    return _fn


def test_nightly_refresh_only_touches_promoted_tickers(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    calls: list[tuple] = []

    def _tour(config, ticker, side="long", **_kw):
        calls.append((ticker, side))
        return _stub_tournament(promoted=True)(config, ticker, side)

    monkeypatch.setattr("berich.training.tournament.train_ticker_tournament", _tour)

    # No promoted model yet -> the ticker is skipped.
    out = ticker_nightly_refresh_job(cfg)
    assert out["refreshed"] == 0
    assert out["skipped"] == 1
    assert calls == []

    # Promote one, and the nightly job now refreshes it.
    _promote_stub(tmp_path, "AAA", "long")
    out = ticker_nightly_refresh_job(cfg)
    assert out["refreshed"] == 1
    assert out["promoted"] == 1
    assert calls == [("AAA", "long")]


def test_initial_sweep_runs_every_ticker_side(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    calls: list[tuple] = []

    def _tour(config, ticker, side="long", **_kw):
        calls.append((ticker, side))
        return _stub_tournament(promoted=False)(config, ticker, side)

    monkeypatch.setattr("berich.training.tournament.train_ticker_tournament", _tour)

    out = ticker_initial_sweep_job(cfg)
    assert out["swept"] == 1
    assert out["promoted"] == 0
    assert out["failed"] == 0
    assert calls == [("AAA", "long")]


def test_hpo_queue_processes_one_asset_at_a_time(tmp_path, monkeypatch):
    # Two tradeable tickers, one side each => 2 pending pairs (no optuna.db => none searched yet).
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save("AAA", _ohlcv())
    store.save("BBB", _ohlcv(seed=2))
    store.save("SPY", _ohlcv(seed=1))
    cfg = Config(data_dir=tmp_path, universes={"us_stocks": ["AAA", "BBB"]})
    cfg.zoo.ticker_sides = ["long"]
    cfg.zoo.ticker_tournament_models = ["lgbm"]
    cfg.zoo.ticker_exit_strategies = ["fixed"]

    calls: list[tuple] = []

    def _record(config, ticker, side, n_trials, strategy):  # noqa: ARG001
        calls.append((ticker, side, strategy))
        return False

    monkeypatch.setattr(jobs_mod, "_hpo_and_tournament", _record)

    # max_assets=1 => only the first pending triple is processed; the rest stay queued.
    out = jobs_mod.ticker_hpo_queue_job(cfg, max_assets=1)
    assert out["processed"] == ["AAA/long/fixed"]
    assert out["remaining"] == 1
    assert calls == [("AAA", "long", "fixed")]


def test_pending_hpo_targets_prioritizes_new_arrivals(tmp_path, monkeypatch):
    # OLD has a long/fixed study (already onboarded); NEW was just added with no studies at all.
    cfg = Config(data_dir=tmp_path, universes={"us_stocks": ["OLD", "NEW"]})
    cfg.zoo.ticker_sides = ["long"]
    cfg.zoo.ticker_exit_strategies = ["fixed", "trailing"]

    monkeypatch.setattr(
        "berich.training.status._hpo_trial_counts",
        lambda _db: {"berich-hpo-OLD-lgbm-long-sharpe": 50},
    )

    pending = jobs_mod._pending_hpo_targets(cfg)
    # OLD/long/fixed already has trials (dropped); the new arrival jumps ahead of OLD's leftover.
    assert pending == [
        ("NEW", "long", "fixed"),
        ("NEW", "long", "trailing"),
        ("OLD", "long", "trailing"),
    ]


def test_initial_sweep_covers_every_exit_strategy(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg.zoo.ticker_exit_strategies = ["fixed", "trailing"]
    calls: list[tuple] = []

    def _record(config, ticker, side, n_trials, strategy):  # noqa: ARG001
        calls.append((ticker, side, strategy))
        return False

    monkeypatch.setattr(jobs_mod, "_hpo_and_tournament", _record)
    out = ticker_initial_sweep_job(cfg)
    # One ticker x one side x two strategies => both strategies swept.
    assert out["swept"] == 2
    assert calls == [("AAA", "long", "fixed"), ("AAA", "long", "trailing")]


def test_hpo_lock_is_exclusive_across_holders(tmp_path):
    cfg = Config(data_dir=tmp_path)
    fd1 = jobs_mod.acquire_hpo_lock(cfg)
    assert fd1 is not None
    # A second holder cannot take it while the first is open (single-driver rule).
    assert jobs_mod.acquire_hpo_lock(cfg) is None
    os.close(fd1)
    # Once released, it can be acquired again.
    fd2 = jobs_mod.acquire_hpo_lock(cfg)
    assert fd2 is not None
    os.close(fd2)


def test_hpo_queue_skips_when_lock_held(tmp_path, monkeypatch):
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save("AAA", _ohlcv())
    store.save("SPY", _ohlcv(seed=1))
    cfg = Config(data_dir=tmp_path, universes={"us_stocks": ["AAA"]})
    cfg.zoo.ticker_sides = ["long"]
    cfg.zoo.ticker_exit_strategies = ["fixed"]

    called = []
    monkeypatch.setattr(jobs_mod, "_hpo_and_tournament", lambda *a: called.append(a) or False)
    held = jobs_mod.acquire_hpo_lock(cfg)  # simulate the sweep service holding the lock
    try:
        out = jobs_mod.ticker_hpo_queue_job(cfg, max_assets=1)
    finally:
        os.close(held)
    assert out["skipped"] == "locked"
    assert called == []  # nothing trained while the lock was held


def test_cmd_train_tournament_single_ticker(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg.model_dump_json())
    monkeypatch.setattr(Config, "load", classmethod(lambda _cls, _p: cfg))

    seen: list[tuple] = []

    def _tour(config, ticker, side="long", *, force=False, **_kw):
        seen.append((ticker, side, force))
        return _stub_tournament(promoted=True)(config, ticker, side)

    monkeypatch.setattr("berich.training.tournament.train_ticker_tournament", _tour)

    args = argparse.Namespace(
        config=str(cfg_path),
        tournament=True,
        all_tickers=False,
        ticker="AAA",
        side="long",
        force=False,
    )
    assert _cmd_train(args) == 0
    assert seen == [("AAA", "long", False)]
