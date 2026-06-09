"""Tests for PSI, feature drift, and the scheduler wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.config import Config
from berich.data.store import OhlcvStore
from berich.monitoring.drift import (
    PSI_SIGNIFICANT,
    DriftReport,
    FeatureDrift,
    feature_drift,
    population_stability_index,
    split_reference_recent,
)
from berich.scheduler import build_scheduler
from berich.scheduler import jobs as jobs_mod
from berich.scheduler.jobs import ticker_drift_monitor_job


def test_psi_identical_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    assert population_stability_index(x, y) < 0.1


def test_psi_shifted_is_significant():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(3, 1, 5000)  # large mean shift
    assert population_stability_index(x, y) >= PSI_SIGNIFICANT


def test_feature_drift_flags_shifted_column():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(0, 1, 3000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 3000), "b": rng.normal(4, 1, 3000)})
    report = feature_drift(ref, cur)
    flags = {f.feature: f.drifted for f in report.features}
    assert flags["b"] is True
    assert flags["a"] is False


def test_should_retrain_threshold():
    drifted = [FeatureDrift(f"f{i}", psi=0.5, ks_pvalue=0.0) for i in range(2)]
    stable = [FeatureDrift(f"g{i}", psi=0.0, ks_pvalue=1.0) for i in range(4)]
    # 2 of 6 drifted == 1/3, which meets the retrain threshold.
    assert DriftReport(features=drifted + stable).should_retrain is True
    # 1 of 6 is below the threshold.
    assert DriftReport(features=drifted[:1] + stable).should_retrain is False


def test_build_scheduler_registers_jobs():
    scheduler = build_scheduler(Config(watchlist=["AAPL"]))
    job_ids = {job.id for job in scheduler.get_jobs()}
    assert job_ids == {
        "daily_paper",
        "intraday_paper",
        "refresh_universe",
        "check_drift",
        "ticker_drift_monitor",
        "ticker_nightly_refresh",
        "ticker_initial_sweep",
        "ticker_hpo_queue",
        "nightly_hpo",
        "weekend_hpo",
        "longshort_signals",
        "refresh_signals",
        "backup",
    }


def test_split_reference_recent():
    feats = pd.DataFrame({"a": range(200)})
    # Too short -> None (not enough reference history).
    assert split_reference_recent(feats.iloc[:100], recent_window=60, min_reference=120) is None
    split = split_reference_recent(feats, recent_window=60, min_reference=120)
    assert split is not None
    reference, recent = split
    assert len(recent) == 60
    assert len(reference) == 140


def _ohlcv_ending(end: pd.Timestamp, n: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    rets = rng.normal(0.0, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


def _drift_job_setup(tmp_path, monkeypatch, df: pd.DataFrame):
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save("AAA", df)
    store.save("SPY", df)
    cfg = Config(data_dir=tmp_path, universes={"us_stocks": ["AAA"]})
    monkeypatch.setattr("berich.signals.service._optimized_tickers", lambda _c: ["AAA"])
    monkeypatch.setattr(jobs_mod, "_has_promoted_model", lambda *_a, **_k: True)
    alerts: list[tuple[list, list]] = []
    monkeypatch.setattr(jobs_mod, "_alert_data_health", lambda s, f: alerts.append((s, f)))
    return cfg, alerts


def test_drift_monitor_no_email_on_healthy_data(tmp_path, monkeypatch):
    # Fresh, moving data -> drift shares are LOGGED but never emailed (cry-wolf post-mortem).
    df = _ohlcv_ending(pd.Timestamp.today().normalize())
    cfg, alerts = _drift_job_setup(tmp_path, monkeypatch, df)
    summary = ticker_drift_monitor_job(cfg)
    assert summary["scanned"] == 1
    assert summary["stale"] == [] and summary["frozen"] == []
    assert not alerts


def test_drift_monitor_alerts_on_stale_cache(tmp_path, monkeypatch):
    # Last bar months old -> the promoted asset is served from a dead feed: actionable email.
    df = _ohlcv_ending(pd.Timestamp.today().normalize() - pd.Timedelta(days=90))
    cfg, alerts = _drift_job_setup(tmp_path, monkeypatch, df)
    summary = ticker_drift_monitor_job(cfg)
    assert summary["stale"] == ["AAA"]
    assert alerts and alerts[0][0][0][0] == "AAA"


def test_drift_monitor_alerts_on_frozen_prices(tmp_path, monkeypatch):
    # Fresh dates but identical closes across the tail -> frozen feed: actionable email.
    df = _ohlcv_ending(pd.Timestamp.today().normalize())
    df.iloc[-5:, df.columns.get_loc("close")] = 123.45
    cfg, alerts = _drift_job_setup(tmp_path, monkeypatch, df)
    summary = ticker_drift_monitor_job(cfg)
    assert summary["frozen"] == ["AAA"]
    assert alerts and alerts[0][1] == ["AAA"]
