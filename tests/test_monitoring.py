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


def test_ticker_drift_monitor_alerts_on_regime_shift(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 320
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # First ~260 bars calm, last 60 a clear volatility/return regime shift -> features drift.
    rets = np.concatenate([rng.normal(0.0, 0.01, n - 60), rng.normal(0.02, 0.08, 60)])
    close = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )
    store = OhlcvStore(tmp_path / "ohlcv")
    store.save("AAA", df)
    store.save("SPY", df)
    cfg = Config(data_dir=tmp_path, universes={"us_stocks": ["AAA"]})

    # Pretend AAA is optimized + promoted, and capture the alert instead of sending email.
    monkeypatch.setattr("berich.signals.service._optimized_tickers", lambda _c: ["AAA"])
    monkeypatch.setattr(jobs_mod, "_has_promoted_model", lambda *_a, **_k: True)
    alerts: list[list[tuple[str, float]]] = []

    def _capture(drifted: list[tuple[str, float]]) -> None:
        alerts.append(drifted)

    monkeypatch.setattr(jobs_mod, "_alert_drifted_assets", _capture)

    summary = ticker_drift_monitor_job(cfg)
    assert summary["scanned"] == 1
    assert summary["drifted"] == 1
    assert "AAA" in summary["tickers"]
    assert alerts and alerts[0][0][0] == "AAA"
