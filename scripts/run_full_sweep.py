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
from berich.training.promotion import reconcile_sweep_fdr
from berich.training.status import _hpo_trial_counts, _hpo_trials_for

# Run the sweep-level FDR reconcile this often (every N tournaments) so the anti-luck demotion
# happens continuously, instead of only at the end of a full 600-combo cycle that may never
# complete before a restart resets it.
_FDR_EVERY = 30

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


def _fdr(config: Config, tag: str) -> None:
    """Best-effort sweep-level FDR reconcile (demote promotions that fail multiple-testing)."""
    try:
        r = reconcile_sweep_fdr(config)
        log.info("FDR %s: %d promoted -> %d demoted", tag, r["promoted_before"], r["demoted"])
    except Exception:
        log.exception("FDR %s failed", tag)


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
    """Perpetual, *smart* retraining loop: cycle through every (ticker, side, strategy) forever.

    It is not a dumb redo-the-same-thing loop. Two things keep it useful, not wasteful:

    - **Fresh data every cycle.** OHLCV is refreshed up front, so each re-fit learns on the latest
      bars — the market is what changed, even when the model "stays the same".
    - **Tapered search.** A new/shallow study gets the full deep HPO (``ticker_initial_hpo_trials``)
      to find good hyperparameters; once a study is already deep it switches to a light top-up
      (``ticker_nightly_hpo_trials``) and mostly just re-fits on fresh data — no pointlessly
      re-searching a converged space, so the GPUs aren't burned on redundant work and the Optuna
      DB grows slowly.

    Priority: un-trained combos are processed FIRST, so a newly added asset is trained ahead of
    re-deepening existing ones. Config is reloaded each cycle, so new tickers / strategies / tuned
    params are picked up automatically (a restart just makes it immediate).

    Runs under systemd (Restart=always); the cross-process lock keeps the scheduler's jobs yielding.
    """
    del config  # reloaded fresh each cycle (below)
    # Reconcile once at startup: the service restarts (Restart=always) reset the cycle counter, so
    # without this the end-of-cycle FDR could never fire on a long first pass.
    _fdr(Config.load("config/berich.yaml"), "startup")
    cycle = 0
    while True:
        cycle += 1
        c0 = time.time()
        config = Config.load("config/berich.yaml")
        deep_trials = config.zoo.ticker_initial_hpo_trials
        topup_trials = config.zoo.ticker_nightly_hpo_trials
        combos = [
            (ticker, side, strategy)
            for ticker in config.tradeable_tickers()
            for side in config.zoo.ticker_sides
            for strategy in config.zoo.ticker_exit_strategies
        ]
        try:
            update_watchlist(config)
            log.info("cycle %d: OHLCV refreshed, %d combos", cycle, len(combos))
        except Exception:
            log.exception("cycle %d: data refresh failed (training on cached bars)", cycle)
        # Priority: un-searched combos (new assets) first, then deepen the rest.
        counts = _hpo_trial_counts(config.optuna_db)
        combos.sort(key=lambda c: _hpo_trials_for(counts, c[0], None, c[1], c[2]) > 0)
        promoted = 0
        for i, (ticker, side, strategy) in enumerate(combos):
            existing = _hpo_trials_for(counts, ticker, None, side, strategy)
            n_trials = deep_trials if existing < deep_trials else topup_trials
            log.info(
                "[cycle %d, %d/%d] start %s/%s/%s (%d trials, %d existing)",
                cycle,
                i + 1,
                len(combos),
                ticker,
                side,
                strategy,
                n_trials,
                existing,
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
            # Sweep-level multiple-testing control every _FDR_EVERY tournaments — the continuous
            # sweep holds the HPO lock perpetually (so the scheduler's FDR job always yields), and a
            # full 600-combo cycle may never complete before a restart. Each tournament already
            # corrected for its own search; this walks back promotions that don't survive
            # Benjamini-Hochberg across the whole promoted set (demoted -> observe tier).
            if (i + 1) % _FDR_EVERY == 0:
                _fdr(config, f"cycle {cycle} @{i + 1}/{len(combos)}")
        _fdr(config, f"cycle {cycle} end")
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
