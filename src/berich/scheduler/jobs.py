"""Scheduled job functions.

Each job is a plain function taking a :class:`~berich.config.Config` so it can be
called directly in tests or wired into APScheduler by the runner. Jobs are
idempotent: re-running a day's refresh/signals overwrites rather than duplicates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from berich.data import update_watchlist
from berich.data.store import OhlcvStore
from berich.datasets import build_dataset
from berich.labeling.triple_barrier import LabelConfig
from berich.monitoring import feature_drift
from berich.signals import SignalStore, generate_signals

if TYPE_CHECKING:
    from berich.config import Config
    from berich.monitoring import DriftReport

logger = logging.getLogger(__name__)

# Number of most-recent samples treated as the "current" window for drift.
DRIFT_CURRENT_SAMPLES = 2000


def refresh_and_signal_job(config: Config) -> int:
    """Refresh the cache, regenerate signals, and persist them. Returns signals saved."""
    update_watchlist(config)
    store = OhlcvStore(config.ohlcv_dir)
    signals = generate_signals(config, store)
    saved = SignalStore(config.db_path).save(signals)
    logger.info("refresh_and_signal: %d signals saved", saved)
    return saved


def check_drift_job(config: Config) -> DriftReport:
    """Compare recent feature distributions to the training era; log if retrain needed."""
    store = OhlcvStore(config.ohlcv_dir)
    label_cfg = LabelConfig(**config.labeling.model_dump())
    dataset = build_dataset(store, config.watchlist, label_cfg)

    n_current = min(DRIFT_CURRENT_SAMPLES, len(dataset) // 2)
    reference = dataset.x.iloc[:-n_current]
    current = dataset.x.iloc[-n_current:]
    report = feature_drift(reference, current)

    level = logging.WARNING if report.should_retrain else logging.INFO
    logger.log(
        level,
        "drift: %d/%d features drifted (%.0f%%); retrain=%s",
        report.n_drifted,
        len(report.features),
        report.share_drifted * 100,
        report.should_retrain,
    )
    return report
