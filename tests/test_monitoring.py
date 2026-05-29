"""Tests for PSI, feature drift, and the scheduler wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from berich.config import Config
from berich.monitoring.drift import (
    PSI_SIGNIFICANT,
    DriftReport,
    FeatureDrift,
    feature_drift,
    population_stability_index,
)
from berich.scheduler import build_scheduler


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
        "retrain_zoo",
        "retrain_asset_models",
        "nightly_hpo",
        "weekend_hpo",
        "longshort_signals",
    }
