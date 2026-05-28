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
from berich.signals import (
    SignalStore,
    generate_signals,
    open_new_trades,
    update_open_trades,
)

if TYPE_CHECKING:
    from berich.config import Config
    from berich.monitoring import DriftReport

logger = logging.getLogger(__name__)

# Number of most-recent samples treated as the "current" window for drift.
DRIFT_CURRENT_SAMPLES = 2000


def daily_paper_job(config: Config) -> dict[str, int]:
    """Daily chain: refresh OHLCV → regenerate signals → roll the paper-trade book.

    Returns a dict with ``signals_saved``, ``trades_opened``, ``trades_closed``. Each
    sub-step is idempotent (signals upsert on ``(date, ticker)``; paper opens skip
    rows already present; paper updates only touch ``status='open'``), so the
    scheduler can re-run it safely and the user can run it manually via
    ``berich paper update`` without colliding.
    """
    update_watchlist(config)
    store = OhlcvStore(config.ohlcv_dir)
    signals = generate_signals(config, store)
    signal_store = SignalStore(config.db_path)
    saved = signal_store.save(signals)
    opened = open_new_trades(config, store, signal_store)
    closed = update_open_trades(config, store)
    logger.info(
        "daily_paper: %d signals saved, %d paper opened, %d paper closed",
        saved,
        opened,
        closed,
    )
    return {"signals_saved": saved, "trades_opened": opened, "trades_closed": closed}


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
