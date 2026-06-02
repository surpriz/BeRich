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

import logging
import sys
import time

from berich.config import Config
from berich.scheduler.jobs import _hpo_and_tournament, _pending_hpo_targets, refresh_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("sweep")


def main() -> int:
    config = Config.load("config/berich.yaml")
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
