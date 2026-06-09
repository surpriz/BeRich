"""Intraday serving + scheduler wiring (Lots B2/B3). GPU-CI only (imports the model zoo)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from berich.config import Config
from berich.data.store import OhlcvStore
from berich.scheduler.runner import build_scheduler
from berich.signals.service import generate_intraday_signals


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config(data_dir=tmp_path, universes={"crypto": ["BTC-USD"]})
    cfg.intraday.enabled = True
    cfg.intraday.tickers = ["BTC-USD"]
    return cfg


def test_scheduler_registers_intraday_job_without_disturbing_daily(config):
    # build_scheduler returns an unstarted scheduler; inspect jobs without starting it.
    scheduler = build_scheduler(config)
    assert scheduler.get_job("intraday_paper") is not None
    assert scheduler.get_job("daily_paper") is not None  # daily job untouched


def test_intraday_job_hourly_trigger(config):
    scheduler = build_scheduler(config)
    job = scheduler.get_job("intraday_paper")
    # Fires every hour (minute=2) — crypto trades 24/7, so no day-of-week restriction.
    assert "minute='2'" in str(job.trigger) or "minute=2" in str(job.trigger)


def test_generate_intraday_signals_empty_without_models(config):
    store = OhlcvStore(config.ohlcv_intraday_dir, interval="1h")
    idx = pd.date_range("2024-01-01", periods=300, freq="1h")
    close = 100 + np.cumsum(np.zeros(300))
    store.save(
        "BTC-USD",
        pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1_000.0},
            index=idx,
        ),
    )
    # No interval-dimensioned models trained yet -> no signals, no crash.
    assert generate_intraday_signals(config, store) == []
