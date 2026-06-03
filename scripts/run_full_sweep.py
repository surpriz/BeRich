"""Drain the per-(ticker, side, exit-strategy) HPO + tournament queue to completion.

Runs every un-searched (ticker, side, exit strategy) triple through the full first-pass HPO
(deep models on the GPU pool) + tournament, refreshing the served signals after each, until
nothing is pending. Resumable: a triple drops off as soon as its Optuna study has >=1 trial,
so re-running continues where it left off. A triple that errors before recording any trial is
given up on (so the loop can't spin forever on a broken asset).

Run it as a long-lived background process while the scheduler's competing hpo_queue is paused:

    systemctl stop berich-scheduler.service
    nohup uv run python scripts/run_full_sweep.py > data/sweep.log 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from berich.config import Config
from berich.data import update_watchlist
from berich.scheduler.jobs import (
    _hpo_and_tournament,
    _pending_hpo_targets,
    acquire_hpo_lock,
    refresh_signals,
)

_LOG_PATH = Path("data/sweep.log")
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        # Own the log file (append, capped) so it works identically whether launched by systemd
        # or by hand, and /ops can read the drainer's activity from data/sweep.log.
        RotatingFileHandler(_LOG_PATH, maxBytes=20_000_000, backupCount=2),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("sweep")


def main() -> int:
    parser = argparse.ArgumentParser(description="BeRich training sweep / continuous retrainer")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Never stop: after draining, perpetually retrain every asset with fresh data and "
        "deeper HPO (keeps the rented machine fully used and models always current).",
    )
    args = parser.parse_args()

    config = Config.load("config/berich.yaml")
    # Single-driver guarantee: hold the cross-process HPO lock for the whole run so the scheduler's
    # HPO jobs yield to us (no Optuna/registry write races). Retry a while in case a scheduler job
    # is mid-triple when we (re)start under systemd.
    lock = None
    for _ in range(60):
        lock = acquire_hpo_lock(config)
        if lock is not None:
            break
        log.info("HPO lock busy (scheduler job running?), retrying in 30s")
        time.sleep(30)
    if lock is None:
        log.error("could not acquire HPO lock after 30 min; exiting")
        return 1
    try:
        return _continuous(config) if args.continuous else _drain(config)
    finally:
        os.close(lock)


def _continuous(config: Config) -> int:
    """Perpetual retraining loop: cycle through EVERY (ticker, side, strategy) forever.

    Each cycle refreshes the latest OHLCV, then re-runs HPO + tournament for every combo — so
    Optuna studies deepen over time (more trials = better hyperparameters), models re-fit on the
    freshest bars, and the guard re-promotes the honest winner. The rented GPUs never idle. Runs
    under systemd (Restart=always); the cross-process lock keeps the scheduler's HPO jobs yielding.
    """
    n_trials = config.zoo.ticker_initial_hpo_trials
    combos = [
        (ticker, side, strategy)
        for ticker in config.tradeable_tickers()
        for side in config.zoo.ticker_sides
        for strategy in config.zoo.ticker_exit_strategies
    ]
    cycle = 0
    while True:
        cycle += 1
        c0 = time.time()
        try:
            update_watchlist(config)
            log.info("cycle %d: OHLCV refreshed", cycle)
        except Exception:
            log.exception("cycle %d: data refresh failed (training on cached bars)", cycle)
        promoted = 0
        for i, (ticker, side, strategy) in enumerate(combos):
            log.info(
                "[cycle %d, %d/%d] start %s/%s/%s",
                cycle,
                i + 1,
                len(combos),
                ticker,
                side,
                strategy,
            )
            t0 = time.time()
            try:
                ok = _hpo_and_tournament(config, ticker, side, n_trials, strategy)
                promoted += int(bool(ok))
                log.info(
                    "done %s/%s/%s in %.0fs promoted=%s",
                    ticker,
                    side,
                    strategy,
                    time.time() - t0,
                    ok,
                )
            except Exception:
                log.exception("retrain failed %s/%s/%s", ticker, side, strategy)
            try:
                refresh_signals(config)
            except Exception:
                log.exception("refresh_signals failed after %s/%s/%s", ticker, side, strategy)
        log.info(
            "cycle %d COMPLETE: %d combos, %d promoted, %.0f min",
            cycle,
            len(combos),
            promoted,
            (time.time() - c0) / 60.0,
        )


def _drain(config: Config) -> int:
    n_trials = config.zoo.ticker_initial_hpo_trials
    giveup: set[tuple[str, str, str]] = set()
    done = 0
    promoted_total = 0
    t_start = time.time()

    while True:
        pending = [t for t in _pending_hpo_targets(config) if t not in giveup]
        if not pending:
            break
        ticker, side, strategy = pending[0]
        log.info(
            "[%d done, %d promoted, %d pending] start %s/%s/%s",
            done,
            promoted_total,
            len(pending),
            ticker,
            side,
            strategy,
        )
        t0 = time.time()
        try:
            promoted = _hpo_and_tournament(config, ticker, side, n_trials, strategy)
            promoted_total += int(bool(promoted))
            log.info(
                "done %s/%s/%s in %.0fs promoted=%s",
                ticker,
                side,
                strategy,
                time.time() - t0,
                promoted,
            )
        except Exception:
            log.exception("FAILED %s/%s/%s", ticker, side, strategy)
        # If the triple is still pending after the attempt (no trial recorded), give up so the
        # loop can make progress instead of retrying a broken asset forever.
        if (ticker, side, strategy) in set(_pending_hpo_targets(config)):
            giveup.add((ticker, side, strategy))
            log.warning("giving up on %s/%s/%s (no trials recorded)", ticker, side, strategy)
        done += 1
        try:
            refresh_signals(config)
        except Exception:
            log.exception("refresh_signals failed after %s/%s/%s", ticker, side, strategy)

    log.info(
        "SWEEP COMPLETE: %d triples processed, %d promoted, %d gave up, %.0f min total",
        done,
        promoted_total,
        len(giveup),
        (time.time() - t_start) / 60.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
